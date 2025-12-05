#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <ctime>

using u8 = unsigned char;

#define CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Definiamo le dimensioni del blocco a compile-time per la Shared Memory
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define RADIUS 1

// La tile in Shared Memory deve contenere il blocco + i bordi
// Dimensione Shared: (16 + 2) x (16 + 2) = 18x18
#define SM_W (BLOCK_DIM_X + 2 * RADIUS)
#define SM_H (BLOCK_DIM_Y + 2 * RADIUS)

// --- KERNEL MODIFICATO (Versione Linear Loading) ---
__global__ void gol_step_shared(u8* src, u8* dst, int width, int height) {
    
    // 1. Definiamo la dimensione totale della tile lineare (18*18 = 324)
    // Usiamo una costante per permettere al compilatore di ottimizzare
    const int SM_SIZE = SM_W * SM_H;

    // 2. Allocazione Shared Memory come array 1D lineare
    // Questo facilita il caricamento collaborativo
    __shared__ u8 tile[SM_SIZE];

    // --- FASE 1: CARICAMENTO LINEARE COLLABORATIVO ---
    // Tutti i thread collaborano per riempire il buffer, senza curarsi della geometria 2D
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x; // ID lineare del thread (0..255)
    int num_threads = blockDim.x * blockDim.y;        // Totale thread (256)

    // Coordinate globali dell'angolo in alto a sinistra della tile (incluso halo)
    int base_gx = blockIdx.x * blockDim.x - RADIUS;
    int base_gy = blockIdx.y * blockDim.y - RADIUS;

    // Ciclo di caricamento con stride
    // I thread coprono l'intera dimensione SM_SIZE (324 elementi)
    for (int i = tid; i < SM_SIZE; i += num_threads) {
        
        // Mappiamo l'indice lineare 'i' alle coordinate 2D relative alla tile
        int sm_y = i / SM_W;
        int sm_x = i % SM_W;

        // Calcoliamo la posizione globale reale da cui leggere
        int global_y = base_gy + sm_y;
        int global_x = base_gx + sm_x;

        // Caricamento coalesced (thread contigui leggono indirizzi contigui)
        if (global_x >= 0 && global_x < width && global_y >= 0 && global_y < height) {
            tile[i] = src[global_y * width + global_x];
        } else {
            tile[i] = 0; // Padding (zero) per i bordi fuori immagine
        }
    }

    // Barriera: aspettiamo che tutti abbiano finito di caricare
    __syncthreads();

    // --- FASE 2: CALCOLO (Logica 2D standard) ---
    
    // Coordinate del thread per l'output
    int lx = threadIdx.x;
    int ly = threadIdx.y;
    int gx = blockIdx.x * blockDim.x + lx;
    int gy = blockIdx.y * blockDim.y + ly;

    // Uscita se fuori dalla griglia
    if (gx >= width || gy >= height) return;

    // Calcoliamo l'indice centrale in Shared Memory (1D)
    // Il thread (0,0) corrisponde all'offset (RADIUS, RADIUS) nella tile
    int center_idx = (ly + RADIUS) * SM_W + (lx + RADIUS);

    // Somma dei vicini usando offset precalcolati (senza if complessi)
    int neighbors_alive = 0;

    // Riga Sopra
    neighbors_alive += tile[center_idx - SM_W - 1];
    neighbors_alive += tile[center_idx - SM_W];
    neighbors_alive += tile[center_idx - SM_W + 1];

    // Riga Corrente (Sinistra e Destra)
    neighbors_alive += tile[center_idx - 1];
    neighbors_alive += tile[center_idx + 1];

    // Riga Sotto
    neighbors_alive += tile[center_idx + SM_W - 1];
    neighbors_alive += tile[center_idx + SM_W];
    neighbors_alive += tile[center_idx + SM_W + 1];

    // Stato attuale
    u8 cell_value = tile[center_idx];

    // Scrittura risultato
    dst[gy * width + gx] = (neighbors_alive == 3) || (cell_value && (neighbors_alive == 2));
}

// host helper
void random_board(u8* board, int width, int height, float alive_prob = 0.2f) {
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            board[y * width + x] = (float(rand()) / RAND_MAX) < alive_prob ? 1 : 0;
}

// Inizializza la griglia con valori 0 o 1 in modo deterministico
void init_random_reproducible(u8* grid, int width, int height, unsigned int seed) {
    
    // Probabilità che sia 0 o 1
    float probability = 0.5;
    // Seed riproducibile
    srand(seed);

    int total_cells = width * height;

    for (int i = 0; i < total_cells; ++i) {
        // Genera un float tra 0.0 e 1.0
        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        
        // Se r è minore della probabilità (es. 0.5), la cella è viva (1), altrimenti morta (0)
        grid[i] = (r < probability) ? 1 : 0;
    }
}

int main(int argc, char** argv) {
    int width = 1024;   // non più 32
    int height = 1024;  // non più 32
    int steps = 15;

    size_t griglia = size_t(width) * height;
    size_t total_bytes = griglia * sizeof(u8);

    u8* h_board = (u8*)malloc(total_bytes);     // host board allocation
    srand((unsigned)time(NULL));
    //random_board(h_board, width, height, 0.15f);
    //initialize_glider(h_board, width);
    init_random_reproducible(h_board, width, height, 42);

    // alloca memoria device
    u8 *d_a, *d_b;
    CHECK(cudaMalloc(&d_a, total_bytes));
    CHECK(cudaMalloc(&d_b, total_bytes));
    CHECK(cudaMemcpy(d_a, h_board, total_bytes, cudaMemcpyHostToDevice));

    // DIMENSIONE DEL BLOCCO: 2D
    // Fare test per capire configurazione migliore e verificare 
    // occupancy tramite nsight compute

    // Dimensioni dei blocchi (Numero di thread) 
    const int BLOCK_SIZE_X = 16;
    const int BLOCK_SIZE_Y = 16;
    /*
    * 256 threads => 8 warp per blocco
    */
    
    // --- DIMENSIONAMENTO DI GRIGLIA E BLOCCHI ---
    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    // Dimensione della griglia 2D calcolata in relazione a:
    //  1. dimensione della griglia (w, h)
    //  2. dimensione dei blocchi (# th)
    dim3 dimGrid(
        (width + dimBlock.x - 1) / dimBlock.x,
        (height + dimBlock.y - 1) / dimBlock.y
    );

    u8* src = d_a;
    u8* dst = d_b;

    // Kernel execution
    for (int s = 0; s < steps; ++s) {
        gol_step_shared<<<dimGrid, dimBlock>>>(src, dst, width, height);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        // traferisco la griglia GPU -> CPU
        CHECK(cudaMemcpy(h_board, dst, total_bytes, cudaMemcpyDeviceToHost)); 

        // Swap buffers
        u8* tmp = src;
        src = dst;
        dst = tmp;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    free(h_board);
}