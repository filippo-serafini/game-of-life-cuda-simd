/*
*       VERSIONE CUDA GRIGLIA 2D BLOCCO 2D (16x16) E OTTIMIZZAZIONI
*       Max grid y-dimension: 65.535
*       Max thread per block: 1024
*       Griglia di gioco: 1024x1024
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <ctime>

// Alias unsigned char
using u8 = unsigned char;

#define CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// -----------------------------------
//              KERNEL
// -----------------------------------
__global__ void gol_step_2d2d(u8* src, u8* dst, int width, int height, int RADIUS) {

    // Calcolo degli indici di cella con indice thread globale (Global Indexing)
    int cell_index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int cell_index_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Controllo per indici oltre la griglia di gioco.
    // Caso in cui il numero di thread eccede quello delle celle
    // per questo blocco sepecifico [Warp Divergence]
    if (cell_index_x >= width || cell_index_y >= height) {
        return;
    }

    int neighbors_alive = 0;
    for (int dy = -RADIUS; dy <= RADIUS; ++dy) 
    {
        for (int dx = -RADIUS; dx <= RADIUS; ++dx) 
        {
            int neighbor_cell_x = cell_index_x + dx;
            int neighbor_cell_y = cell_index_y + dy;

            // Conto il contributo solo delle celle appartenenti alla griglia di gioco.
            // Per le celle di padding non aggiorno il conteggio (vale 0) [Warp Divergence]
            if(neighbor_cell_x >= 0 && neighbor_cell_x < width 
                && neighbor_cell_y >= 0 && neighbor_cell_y < height)
                {
                    // Accesso alla memoria globale *src*
                    neighbors_alive += src[neighbor_cell_y * width + neighbor_cell_x];
                }
        }
    }

    // Calcolo dell'indice per l'accesso in memoria 1D [row-major]
    int cell_mem_idx = cell_index_y * width + cell_index_x;

    u8 cell_value = src[cell_mem_idx];
    // Escludo la cella centrale dal conteggio
    neighbors_alive -= cell_value; 
    u8 cell_result = 0;
        
    // Risolta warp divergence qui
    cell_result = (neighbors_alive == 3) || (cell_value && (neighbors_alive == 2));

    // Salvataggio del risultato in memoria globale *dst*
    dst[cell_mem_idx] = cell_result;
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
    int width = 1024;
    int height = 1024;
    int steps = 15;
    int radius = 1;

    // Griglia di gioco
    size_t griglia = size_t(width) * height;
    size_t bytes = griglia * sizeof(u8);

    // Alloco memoria sull'host
    u8* host_board = (u8*)malloc(bytes);
    
    // Inizializzazione random riproducibile
    srand((unsigned)time(NULL));
    init_random_reproducible(host_board, width, height, 42);

    // Alloco memoria sul device
    u8 *dev_a, *dev_b;
    CHECK(cudaMalloc(&dev_a, bytes));
    CHECK(cudaMalloc(&dev_b, bytes));
    // Copio la griglia sul device
    CHECK(cudaMemcpy(dev_a, host_board, bytes, cudaMemcpyHostToDevice));

    // DIMENSIONE DEL BLOCCO: 2D
    // blocchi 16x16 => 256 thread (8 warp)
    const int BLOCK_SIZE_X = 16;
    const int BLOCK_SIZE_Y = 16;
    
    // --- DIMENSIONAMENTO DI GRIGLIA E BLOCCHI ---
    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    // Dimensione della griglia 2D calcolata in relazione a:
    //  1. dimensione della griglia (w, h)
    //  2. dimensione dei blocchi (# th)
    dim3 dimGrid(
        (width + dimBlock.x - 1) / dimBlock.x,
        (height + dimBlock.y - 1) / dimBlock.y
    );

    u8* src = dev_a;
    u8* dst = dev_b;

    // Lancio del kernel per il numero di generazioni
    for (int s = 0; s < steps; ++s) {
        gol_step_2d2d<<<dimGrid, dimBlock>>>(src, dst, width, height, radius);
        CHECK(cudaGetLastError());
        
        // Sincronizzazione Host Device necessaria 
        // prima di poter eseguire la prossima generazione
        CHECK(cudaDeviceSynchronize());
        
        // Swap dei buffer
        u8* tmp = src;
        src = dst;
        dst = tmp;
    }

    // Liberazione memoria finale host e device
    cudaFree(dev_a);
    cudaFree(dev_b);
    free(host_board);
}