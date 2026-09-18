// ==========================================================
//   Minimal CUDA + OpenGL Interop - Game of Life Viewer
// ==========================================================

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <stdio.h>
#include <cstdio>
#include <cstdlib>
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
    int cell_index_x = threadIdx.x;
    int cell_index_y = threadIdx.y;

    if (cell_index_x >= width || cell_index_y >= height) return;

    int neighbors_alive = 0;
    for (int dy = -RADIUS; dy <= RADIUS; ++dy)
    {
        for (int dx = -RADIUS; dx <= RADIUS; ++dx)
        {
            int neighbor_cell_x = cell_index_x + dx;
            int neighbor_cell_y = cell_index_y + dy;

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

    if (cell_value) // Vivo
        cell_result = (neighbors_alive == 2 || neighbors_alive == 3) ? 1 : 0;
    else // Morto
        cell_result = (neighbors_alive == 3) ? 1 : 0;

    dst[cell_index_y * width + cell_index_x] = cell_result;
}

// ----------------------------------------------------------
// CUDA kernel: write board into OpenGL PBO (RGB)
// ----------------------------------------------------------
__global__ void renderKernel(uchar4* pbo, u8* board, int width, int height) {
    int cell_index_x = threadIdx.x;
    int cell_index_y = threadIdx.y;

    if (cell_index_x >= width || cell_index_y >= height) return;
    int idx = cell_index_y*width + cell_index_x;
    u8 c = board[idx]*255;

    pbo[idx] = make_uchar4(c, c, c, 255);
}

// host helper
void random_board(u8* board, int width, int height, float alive_prob = 0.2f) {
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            board[y * width + x] = (float(rand()) / RAND_MAX) < alive_prob ? 1 : 0;
}

// Inizializza la griglia con un Glider usando char* e una dimensione "padded"
void initialize_glider(u8* board, int width) {

    // Inizializzazione a 0
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
    const int width = 32;
    const int height = 32;
    int steps = 16;
    const int radius = 1;

    // ------------------------------------------------------
    // Init OpenGL Window
    // ------------------------------------------------------
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }

    GLFWwindow* win = glfwCreateWindow(width, height, "CUDA Game of Life", NULL, NULL);
    if (!win) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(win);

    // Inizializza glad (necessario se usi glad)
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        fprintf(stderr, "Failed to initialize GLAD\n");
        glfwDestroyWindow(win);
        glfwTerminate();
        return -1;
    }

    // ------------------------------------------------------
    // Create OpenGL PBO
    // ------------------------------------------------------
    GLuint pbo = 0;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    size_t griglia = size_t(width) * height;
    size_t board_bytes = griglia * sizeof(u8);
    size_t pbo_size = griglia * 4 * sizeof(u8); // RGBA

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, (GLsizeiptr)pbo_size, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Register PBO with CUDA
    cudaGraphicsResource* cuda_pbo = nullptr;
    CHECK(cudaGraphicsGLRegisterBuffer(&cuda_pbo, pbo, cudaGraphicsMapFlagsWriteDiscard));

    // host board allocation and initialization
    u8* h_board = (u8*)malloc(board_bytes);
    if (!h_board) {
        fprintf(stderr, "Host allocation failed\n");
        cudaGraphicsUnregisterResource(cuda_pbo);
        glDeleteBuffers(1, &pbo);
        glfwDestroyWindow(win);
        glfwTerminate();
        return -1;
    }
    srand((unsigned)time(NULL));
    // Modifica qui per scegliere tra glider o random
    initialize_glider(h_board, width);
    //random_board(h_board, width, height, 0.3f);

    // alloca memoria device
    u8 *d_a = nullptr, *d_b = nullptr;
    CHECK(cudaMalloc(&d_a, board_bytes));
    CHECK(cudaMalloc(&d_b, board_bytes));
    CHECK(cudaMemcpy(d_a, h_board, board_bytes, cudaMemcpyHostToDevice));

    // single grid (1D) and single block (2D)
    dim3 dimBlock(width, height);
    dim3 dimGrid(1);

    u8* src = d_a;
    u8* dst = d_b;

    // ------------------------------------------------------
    // Main Loop
    // ------------------------------------------------------
    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();

        // --- 1. GAME OF LIFE STEP ---
        gol_step_naive<<<dimGrid, dimBlock>>>(src, dst, width, height, radius);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        // swap buffers
        u8* tmp = src;
        src = dst;
        dst = tmp;

        // --- 2. MAP PBO FOR CUDA ---
        CHECK(cudaGraphicsMapResources(1, &cuda_pbo, 0));
        uchar4* pbo_dev_ptr = nullptr;
        size_t mapped_size = 0;
        CHECK(cudaGraphicsResourceGetMappedPointer((void**)&pbo_dev_ptr, &mapped_size, cuda_pbo));

        // --- 3. RENDER TO PBO ---
        renderKernel<<<dimGrid, dimBlock>>>(pbo_dev_ptr, src, width, height);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        // --- 4. UNMAP PBO ---
        CHECK(cudaGraphicsUnmapResources(1, &cuda_pbo, 0));

        // --- 5. DRAW FULLSCREEN PIXELS ---
        glClear(GL_COLOR_BUFFER_BIT);

        glRasterPos2i(0, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        glfwSwapBuffers(win);
    }

    // ------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------
    cudaGraphicsUnregisterResource(cuda_pbo);
    glDeleteBuffers(1, &pbo);

    glfwDestroyWindow(win);
    glfwTerminate();

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    free(h_board);

    return 0;
}