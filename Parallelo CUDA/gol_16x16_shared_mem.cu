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

__global__ void gol_step_shared(u8* src, u8* dst, int width, int height) {
    
    // Allocazione Shared Memory STATICA
    __shared__ u8 tile[SM_H][SM_W];

    // Coordinate globali del thread (per leggere da Global Memory)
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    int global_idx = global_y * width + global_x; // row major

    // Coordinate locali nel blocco (0..15)
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;

    // Coordinate locali nella Shared Memory
    // Aggiungo RADIUS per considerare in posizione 0 e
    // blockDim.y+1 le righe/colonne di padding (zeri)
    // -> riga 0 local = riga 1 shared
    int shared_x = local_x + RADIUS;
    int shared_y = local_y + RADIUS;

    // --- CARICAMENTO IN SHARED MEMORY ---
    
    // Caricamento della cella centrale (propria del thread)
    /*
       Ottima ottimizzazione degli accessi => ogni thread accede alla sua cella in maniera coalescente. 
       Il thread 0 accede a x, il thread 1 accede a x+1 ecc...
       Essendo 32 thread x warp => ogni warp chiede 32 byte => un'unica transazione
       Il mem controller vede questa cosa e raccoglie tutte le richieste del warp in 
       una sola transazione da 32
    */
    // Controllo bounds globali
    if (global_x < width && global_y < height) {
        tile[shared_y][shared_x] = src[global_y * width + global_x];
    } else {
        tile[shared_y][shared_x] = 0; // Padding esterno nullo
    }
    // Warp divergence ma non di troppo impatto. Abbiamo stesso
    // numero di celle e thread => non ho thread di riempimento idle nei warp.

    // B. Caricamento dei bordi (Halo/Ghost cells)
    // I thread sui bordi del blocco caricano anche i vicini esterni al blocco.
    
    // Halo Superiore
    // riga 0 -> o è l'ultima riga del blocco precedente
    // o è riga con tutti 0 di zero-padding
    /*
        Ottimizzazione buonina => solo i thread che accedono alla riga 0 eseguono questa istruzione ad indirizzi contigui (coalesced):
        con blocchi 16x16 => 16 thread (metà warp) => 16 byte.
        In questo modo solo 16 dei 32 byte richiesti saranno utilizzati.
        Ma comunque il dato sarà già presente per il blocco che detiene quella riga in L2 (più veloce della DRAM)
    */
    if (local_y < RADIUS) { 
        int load_y = global_y - RADIUS; // riga precedente
        if (load_y >= 0 && global_x < width) // Check bounds
            tile[shared_y - RADIUS][shared_x] = src[load_y * width + global_x];
        else
            tile[shared_y - RADIUS][shared_x] = 0;
    }

    // Halo Inferiore
    // ultima riga del blocco -> mi serve la prima del blocco dopo
    // o è riga con tutti 0 di zero-padding
    /*
        Ottimizzazione buonina => solo i thread che accedono all'ultima riga del blocco eseguono questa istruzione ad indirizzi contigui (coalesced):
        con blocchi 16x16 => 16 thread (metà warp) => 16 byte.
        In questo modo solo 16 dei 32 byte richiesti saranno utilizzati.
        Ma comunque il dato sarà già presente per il blocco che detiene quella riga in L2 (più veloce della DRAM)
    */
    if (local_y >= blockDim.y - RADIUS) {
        int load_y = global_y + RADIUS;
        if (load_y < height && global_x < width)
            tile[shared_y + RADIUS][shared_x] = src[load_y * width + global_x];
        else
            tile[shared_y + RADIUS][shared_x] = 0;
    }

    // Halo Sinistro
    // prima colonna del blocco -> mi serve l'ultima del blocco prima
    // o è colonna con tutti 0 di zero-padding
    if (local_x < RADIUS) {
        int load_x = global_x - RADIUS;
        if (load_x >= 0 && global_y < height)
            tile[shared_y][shared_x - RADIUS] = src[global_y * width + load_x];
        else
            tile[shared_y][shared_x - RADIUS] = 0;
    }

    // Halo Destro
    // ultima colonna del blocco -> mi serve la prima del blocco dopo
    // o è colonna con tutti 0 di zero-padding
    if (local_x >= blockDim.x - RADIUS) {
        int load_x = global_x + RADIUS;
        if (load_x < width && global_y < height)
            tile[shared_y][shared_x + RADIUS] = src[global_y * width + load_x];
        else
            tile[shared_y][shared_x + RADIUS] = 0;
    }

    // Halo Angoli per celle angolari del blocco
    if (local_x < RADIUS && local_y < RADIUS) { // Top-Left
        int load_y = global_y - RADIUS; int load_x = global_x - RADIUS;
        tile[shared_y - RADIUS][shared_x - RADIUS] = (load_x >= 0 && load_y >= 0) ? src[load_y * width + load_x] : 0;
    }
    if (local_x >= blockDim.x - RADIUS && local_y < RADIUS) { // Top-Right
        int load_y = global_y - RADIUS; int load_x = global_x + RADIUS;
        tile[shared_y - RADIUS][shared_x + RADIUS] = (load_x < width && load_y >= 0) ? src[load_y * width + load_x] : 0;
    }
    if (local_x < RADIUS && local_y >= blockDim.y - RADIUS) { // Bottom-Left
        int load_y = global_y + RADIUS; int load_x = global_x - RADIUS;
        tile[shared_y + RADIUS][shared_x - RADIUS] = (load_x >= 0 && load_y < height) ? src[load_y * width + load_x] : 0;
    }
    if (local_x >= blockDim.x - RADIUS && local_y >= blockDim.y - RADIUS) { // Bottom-Right
        int load_y = global_y + RADIUS; int load_x = global_x + RADIUS;
        tile[shared_y + RADIUS][shared_x + RADIUS] = (load_x < width && load_y < height) ? src[load_y * width + load_x] : 0;
    }

    // BARRIERA DI SINCRONIZZAZIONE
    // Necessaria per assicurare che tutto il blocco abbia caricato i dati in Shared Mem
    __syncthreads();

    // --- FASE 2: CALCOLO (Leggendo SOLO da Shared Memory) ---
    
    // Se siamo fuori dalla griglia reale, usciamo (dopo il sync, o i thread attivi aspetterebbero all'infinito quelli usciti prima)
    if (global_x >= width || global_y >= height) return;

    int neighbors_alive = 0;
    
    // Loop ottimizzato (unrolling manuale spesso aiuta, ma il compilatore è bravo)
    for (int dy = -RADIUS; dy <= RADIUS; ++dy) {
        for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
            // Leggo direttamente dalla cache veloce (tile)
            neighbors_alive += tile[shared_y + dy][shared_x + dx];
        }
    }

    u8 cell_value = tile[shared_y][shared_x];
    neighbors_alive -= cell_value; // Rimuovo self

    // Aggiornamento della griglia finale
    dst[global_idx] = (neighbors_alive == 3) || (cell_value && (neighbors_alive == 2));
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
    *   256 threads => 8 warp per blocco
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

        // Swap buffers
        u8* tmp = src;
        src = dst;
        dst = tmp;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    free(h_board);
}