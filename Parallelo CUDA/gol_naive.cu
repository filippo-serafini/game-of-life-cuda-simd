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

// -------------------- KERNEL NAIVE --------------------
__global__ void gol_step_naive(u8* src, u8* dst, int width, int height, int RADIUS) {
    
    // Non ottimale, solo 1024 celle possibili => 32x32 griglia
    // Un blocco solo => un solo SM attivo
    int cell_index_x = threadIdx.x;
    int cell_index_y = threadIdx.y;

    int neighbors_alive = 0;
    for (int dy = -RADIUS; dy <= RADIUS; ++dy) 
    {
        for (int dx = -RADIUS; dx <= RADIUS; ++dx) 
        {
            int neighbor_cell_x = cell_index_x + dx;
            int neighbor_cell_y = cell_index_y + dy;

            // Conto il contributo solo delle celle appartenenti alla griglia (0 o 1)
            // Per le celle di padding non aggiorno il conteggio (vale 0)
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

// Inizializza la griglia con un Glider usando char* e una dimensione "padded"
void initialize_glider(u8* board, int width) {

    for (int y = 0; y < width; ++y)
        for (int x = 0; x < width; ++x)
            board[y * width + x] = 0;

    int r = 10;
    int c = 10;
    board[r * width + c + 1]         = 1;
    board[(r + 1) * width + c + 2]   = 1;
    board[(r + 2) * width + c]       = 1;
    board[(r + 2) * width + c + 1]   = 1;
    board[(r + 2) * width + c + 2]   = 1;
}

int main(int argc, char** argv) {
    int width = 32;
    int height = 32;
    int steps = 15;
    int radius = 1;

    size_t griglia = size_t(width) * height;
    size_t bytes = griglia * sizeof(u8);

    u8* h_board = (u8*)malloc(bytes); // host board allocation
    srand((unsigned)time(NULL));
    //random_board(h_board, width, height, 0.15f);
    init_random_reproducible(h_board, width, height, 42);

    // alloca memoria device
    u8 *d_a, *d_b;
    CHECK(cudaMalloc(&d_a, bytes));
    CHECK(cudaMalloc(&d_b, bytes));
    CHECK(cudaMemcpy(d_a, h_board, bytes, cudaMemcpyHostToDevice));

    // single grid (1D) and single block (2D) 
    // block thread count limited to 1024
    dim3 dimBlock(width, height);
    dim3 dimGrid(1);

    u8* src = d_a;
    u8* dst = d_b;

    // Kernel execution
    for (int s = 0; s < steps; ++s) {
        gol_step_naive<<<dimGrid, dimBlock>>>(src, dst, width, height, radius);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        // traferisco la griglia GPU -> CPU
        CHECK(cudaMemcpy(h_board, dst, bytes, cudaMemcpyDeviceToHost)); 

        // Swap buffers
        u8* tmp = src;
        src = dst; 
        dst = tmp;

    }

    cudaFree(d_a);
    cudaFree(d_b);
    free(h_board);
}