/*
*       VERSIONE CUDA IMPL. NAIVE
*       Griglia: (1,1,1)
*       Blocco:  (32,32,1)
*       Griglia di gioco: 32x32
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
__global__ void gol_step_naive(u8* src, u8* dst, int width, int height, int RADIUS) {
    
    // Indici di cella corrispondono a indici di thread nel blocco
    int cell_index_x = threadIdx.x;
    int cell_index_y = threadIdx.y;

    int neighbors_alive = 0;
    for (int dy = -RADIUS; dy <= RADIUS; ++dy) 
    {
        for (int dx = -RADIUS; dx <= RADIUS; ++dx) 
        {
            int neighbor_cell_x = cell_index_x + dx;
            int neighbor_cell_y = cell_index_y + dy;

            // Conto il contributo solo delle celle appartenenti alla griglia di gioco
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

    u8 cell_value = src[cell_index_y * width + cell_index_x];
    // Escludo la cella centrale dal conteggio
    neighbors_alive -= cell_value; 
    u8 cell_result = 0;
        
    // [Warp divergence]
    if (cell_value)
        cell_result = (neighbors_alive == 2 || neighbors_alive == 3) ? 1 : 0;
    else
        cell_result = (neighbors_alive == 3) ? 1 : 0;

    // Salvataggio del risultato in memoria globale *dst*
    dst[cell_index_y * width + cell_index_x] = cell_result;
}

// Inizializza la griglia con valori 0 o 1 tramite seed riproducibile
void init_random_reproducible(u8* grid, int width, int height, unsigned int seed) {
    
    float probability = 0.5;
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
    int width = 32;
    int height = 32;
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

    // Griglia 1d, 1 blocco 32x32
    dim3 dimBlock(width, height);
    dim3 dimGrid(1);

    u8* src = dev_a;
    u8* dst = dev_b;

    // Lancio del kernel per il numero di generazioni
    for (int s = 0; s < steps; ++s) {
        gol_step_naive<<<dimGrid, dimBlock>>>(src, dst, width, height, radius);
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