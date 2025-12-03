/*
*       Implementazione con griglia 2D e bloccchi 2D
*       In questo modo abbiamo a disposizione molti più thread:
*       Max grid y-dimension: 65.535
*       Max thread per block: 1024
*       => # threads =- 100M threads
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <ctime>

using u8 = unsigned char; // alias per unsigned char

#define CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// -------------------- KERNEL 2d Grid/Block --------------------
__global__ void gol_step_2d2d(u8* src, u8* dst, int width, int height, int RADIUS) {
    
    // Con griglia 2D e blocco 2D posso considerare griglie molto più grandi:
    // blocchi di dim3: 32x23
    // griglia di dim3: [(dataSizeX + blockSizeX - 1) / blockSizeX, 
    //                    (dataSizey + blockSizeY -1) / blockSizeY, 1]
    
    // Calcolo degli indici di cella con indice thread globale (Global Indexing)
    int cell_index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int cell_index_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Controllo se l'indice va oltre i limiti della griglia
    // nel caso in cui il numero di thread eccede quello delle celle
    // per questo blocco sepecifico
    if (cell_index_x >= width || cell_index_y >= height) {
        return; 
    }

    // Calcolo dell'indice 1D per l'accesso in memoria
    int cell_mem_idx = cell_index_y * width + cell_index_x;

    int neighbors_alive = 0;
    for (int dy = -RADIUS; dy <= RADIUS; ++dy) 
    {
        for (int dx = -RADIUS; dx <= RADIUS; ++dx) 
        {
            int neighbor_cell_x = cell_index_x + dx;
            int neighbor_cell_y = cell_index_y + dy;

            // Conto il contributo solo delle celle appartenenti alla griglia (0 o 1)
            // Per le celle di padding non aggiorno il conteggio (vale 0)

            // Warp divergence !!
            if(neighbor_cell_x >= 0 && neighbor_cell_x < width 
                && neighbor_cell_y >= 0 && neighbor_cell_y < height)
                {
                    neighbors_alive += src[neighbor_cell_y * width + neighbor_cell_x];
                }
        }
    }
    u8 cell_value = src[cell_index_y * width + cell_index_x];
    neighbors_alive -= cell_value; // Escludo la cella centrale dal conteggio
    u8 cell_result = 0;
        
    // Warp divergence!!
    if (cell_value) // Vivo
        cell_result = (neighbors_alive == 2 || neighbors_alive == 3) ? 1 : 0;
    else // Morto
        cell_result = (neighbors_alive == 3) ? 1 : 0;

    dst[cell_index_y * width + cell_index_x] = cell_result;
}

// host helper
void random_board(u8* board, int width, int height, float alive_prob = 0.2f) {
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            board[y * width + x] = (float(rand()) / RAND_MAX) < alive_prob ? 1 : 0;
}

// Inizializza la griglia con un Glider
void initialize_glider(u8* board, int width) {

    // Inizializzazione a 0
    for (int y = 0; y < width; ++y)
        for (int x = 0; x < width; ++x)
            board[y * width + x] = 0;

    // Glider
    int r = 10;
    int c = 10;
    board[r * width + c + 1]         = 1;
    board[(r + 1) * width + c + 2]   = 1;
    board[(r + 2) * width + c]       = 1;
    board[(r + 2) * width + c + 1]   = 1;
    board[(r + 2) * width + c + 2]   = 1;
}

int main(int argc, char** argv) {
    int width = 1024;   // non più 32
    int height = 1024;  // non più 32
    int steps = 15;
    int radius = 1;

    size_t griglia = size_t(width) * height;
    size_t total_bytes = griglia * sizeof(u8);

    u8* h_board = (u8*)malloc(total_bytes);     // host board allocation
    srand((unsigned)time(NULL));
    //random_board(h_board, width, height, 0.15f);
    initialize_glider(h_board, width);

    // alloca memoria device
    u8 *d_a, *d_b;
    CHECK(cudaMalloc(&d_a, total_bytes));
    CHECK(cudaMalloc(&d_b, total_bytes));
    CHECK(cudaMemcpy(d_a, h_board, total_bytes, cudaMemcpyHostToDevice));

    // DIMENSIONE DEL BLOCCO: 2D
    // Fare test per capire configurazione migliore e verificare 
    // occupancy tramite nsight compute

    // Primo test con blocchi 32x32 (massimo) => 1024 th per blocco
    const int BLOCK_SIZE_X = 32;
    const int BLOCK_SIZE_Y = 32;
    
    // --- DIMENSIONAMENTO DI GRIGLIA E BLOCCHI ---
    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    // Dimensione della griglia 2D calcolata in relazione a:
    //  1. dimensione della griglia (w, h)
    //  2. dimensione dei blocchi (# th)
    dim3 dimGrid(
        (width + dimBlock.x - 1) / dimBlock.x,
        (width * dimBlock.y - 1) / dimBlock.y
    );

    u8* src = d_a;
    u8* dst = d_b;

    if (width <= 64 && height <= 64) // Stampa solo griglie di dim ragionevoli (max 64x64)
    {
        // Stampa della griglia iniziale
        for (int row = 0; row < width; ++row) {
            for (int col = 0; col < width; ++col)
                putchar(h_board[row * width + col] ? '#' : '.');
            putchar('\n');
        }
        putchar('\n');
    }
    
    // Kernel execution
    for (int s = 0; s < steps; ++s) {
        gol_step_2d2d<<<dimGrid, dimBlock>>>(src, dst, width, height, radius);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        // traferisco la griglia GPU -> CPU
        CHECK(cudaMemcpy(h_board, dst, total_bytes, cudaMemcpyDeviceToHost)); 

        if (width <= 64 && height <= 64) // Stampa solo griglie di dim ragionevoli (max 64x64)
        {
            // Stampa della griglia iniziale
            for (int row = 0; row < width; ++row) {
                for (int col = 0; col < width; ++col)
                    putchar(h_board[row * width + col] ? '#' : '.');
                putchar('\n');
            }
            putchar('\n');
        }

        // Swap buffers
        u8* tmp = src;
        src = dst;
        dst = tmp;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    free(h_board);
}