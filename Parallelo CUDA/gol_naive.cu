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

// Kernel naive: singolo blocco
// i thread elaborano più celle usando striding
__global__ void gol_step_naive(const u8* __restrict__ src, u8* __restrict__ dst, int width, int height) {
    size_t total_cells = size_t(width) * size_t(height);
    int tId = blockIdx.x * blockDim.x + threadIdx.x; // blockIdx.x == 0 al momento del lancio
    int stride = blockDim.x; // siccome gridDim == 1

    for (size_t index = tId; index < total_cells; index += stride) {
        int col = index % width; // calcola coordinate 2D della colonna
        int row = index / width; // calcola coordinate 2D della riga

        int neighbors = 0;
        // esamina 3x3 neighborhood attorno alla cella corrente
        for (int dy = -1; dy <= 1; ++dy) {
            int neighbor_row = row + dy;
            if (neighbor_row < 0 || neighbor_row >= height) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int neighbor_col = col + dx;
                if ( (neighbor_col < 0 || neighbor_col >= width) || (dx == 0 && dy == 0) ) continue;
                neighbors += src[neighbor_row * width + neighbor_col];
            }
        }

        u8 cell = src[row * width + col];
        u8 cell_out = 0;
        if (cell)
            cell_out = (neighbors == 2 || neighbors == 3) ? 1 : 0;
        else
            cell_out = (neighbors == 3) ? 1 : 0;

        dst[index] = cell_out;
    }
}

// host helper
void random_board(u8* board, int width, int height, float alive_prob = 0.2f) {
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            board[y * width + x] = (float(rand()) / RAND_MAX) < alive_prob ? 1 : 0;
}

// Inizializza la griglia con un Glider usando char* e una dimensione "padded"
void initialize_glider(u8* board) {
    int const PADDED_SIZE = 64;
    int r = 10;
    int c = 10;
    board[r * PADDED_SIZE + c + 1]         = 1;
    board[(r + 1) * PADDED_SIZE + c + 2]   = 1;
    board[(r + 2) * PADDED_SIZE + c]       = 1;
    board[(r + 2) * PADDED_SIZE + c + 1]   = 1;
    board[(r + 2) * PADDED_SIZE + c + 2]   = 1;
}

int main(int argc, char** argv) {
    int colonne = 32;
    int righe = 32;
    int steps = 200;

    if (argc >= 3) { colonne = atoi(argv[1]); righe = atoi(argv[2]); }
    if (argc >= 4) steps = atoi(argv[3]);

    size_t celle = size_t(colonne) * righe;
    size_t bytes = celle * sizeof(u8);

    u8* h_board = (u8*)malloc(bytes); // host board allocation
    srand((unsigned)time(NULL));
    random_board(h_board, colonne, righe, 0.15f);
    //initialize_glider(h_board);

    // alloca memoria device
    u8 *d_a, *d_b;
    CHECK(cudaMalloc(&d_a, bytes));
    CHECK(cudaMalloc(&d_b, bytes));
    CHECK(cudaMemcpy(d_a, h_board, bytes, cudaMemcpyHostToDevice));

    // single grid (1) and single block; block thread count limited to 1024
    int threads = (int) ((celle < 1024) ? celle : 1024);
    if (threads < 1) threads = 1;
    dim3 block(threads);
    dim3 grid(1);

    u8* src = d_a;
    u8* dst = d_b;
    
    // esegue steps iterazioni
    for (int s = 0; s < steps; ++s) {
        gol_step_naive<<<grid, block>>>(src, dst, colonne, righe);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        u8* tmp = src; src = dst; dst = tmp;
    }

    CHECK(cudaMemcpy(h_board, src, bytes, cudaMemcpyDeviceToHost));

    // stampa board
    for (int row = 0; row < righe; ++row) {
        for (int col = 0; col < colonne; ++col)
            putchar(h_board[row * colonne + col] ? 'O' : '.');
        putchar('\n');
    }

    cudaFree(d_a);
    cudaFree(d_b);
    free(h_board);
}