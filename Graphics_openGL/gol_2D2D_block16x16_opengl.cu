/*
*       Implementazione con griglia 2D e bloccchi 2D
*       In questo modo abbiamo a disposizione molti più thread:
*       Max grid y-dimension: 65.535
*       Max thread per block: 1024
*       => # threads =- 100M threads
*/

// include per grafica
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include <cuda_runtime.h>
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

// -------------------- KERNEL 2d Grid/Block --------------------
__global__ void gol_step_2d2d(u8* src, u8* dst, int width, int height, int RADIUS) {

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
    u8 cell_value = src[cell_mem_idx];
    neighbors_alive -= cell_value; // Escludo la cella centrale dal conteggio
    u8 cell_result = 0;
        
    // RISOLTA warp divergence
    cell_result = (neighbors_alive == 3) || (cell_value && (neighbors_alive == 2));

    dst[cell_mem_idx] = cell_result;
}

// -------------------- RENDER KERNEL (writes RGBA to PBO) --------------------
__global__ void renderKernel(uchar4* pbo, u8* board, int width, int height) {
    int cell_index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int cell_index_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (cell_index_x >= width || cell_index_y >= height) return;
    int idx = cell_index_y * width + cell_index_x;
    u8 c = board[idx] ? 255 : 0;

    pbo[idx] = make_uchar4(c, c, c, 255);
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
    //dimensioni griglia
    int width = 1024;   // non più 32
    int height = 1024;  // non più 32
    //int steps = 15; continuo fino a chiusura finestra
    int radius = 1;

    size_t griglia = size_t(width) * height;
    size_t total_bytes = griglia * sizeof(u8);
    size_t pbo_bytes = griglia * sizeof(uchar4); // RGBA

    // Inizializza GLFW + window
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }

    GLFWwindow* win = glfwCreateWindow(width, height, "CUDA Game of Life (2D/2D + OpenGL)", NULL, NULL);
    if (!win) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(win);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        fprintf(stderr, "Failed to initialize GLAD\n");
        glfwDestroyWindow(win);
        glfwTerminate();
        return -1;
    }

    // Crea PBO
    GLuint pbo = 0;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, (GLsizeiptr)pbo_bytes, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Registerra PBO con CUDA
    cudaGraphicsResource* cuda_pbo = nullptr;
    CHECK(cudaGraphicsGLRegisterBuffer(&cuda_pbo, pbo, cudaGraphicsMapFlagsWriteDiscard));

    // Alloca memoria host
    u8* h_board = (u8*)malloc(total_bytes);     // host board allocation
    if (!h_board) {
        fprintf(stderr, "Host allocation fallita\n");
        return -1;
    }
    srand((unsigned)time(NULL));
    // Modifica qui per scegliere tra glider o random
    random_board(h_board, width, height, 0.15f);
    //initialize_glider(h_board, width);

    // Alloca memoria device
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
        (width + dimBlock.y - 1) / dimBlock.y
    );

    u8* src = d_a;
    u8* dst = d_b;
    
    // Main loop: compute step -> map PBO -> render -> unmap -> draw
    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();

        // 1) Game of Life step
        gol_step_2d2d<<<dimGrid, dimBlock>>>(src, dst, width, height, radius);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        // swap src/dst
        u8* tmp = src;
        src = dst;  
        dst = tmp;

        // 2) Map PBO for CUDA and get pointer
        CHECK(cudaGraphicsMapResources(1, &cuda_pbo, 0));
        uchar4* pbo_dev_ptr = nullptr;
        size_t mapped_size = 0;
        CHECK(cudaGraphicsResourceGetMappedPointer((void**)&pbo_dev_ptr, &mapped_size, cuda_pbo));

        // 3) Render board -> PBO
        renderKernel<<<dimGrid, dimBlock>>>(pbo_dev_ptr, src, width, height);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        // 4) Unmap PBO
        CHECK(cudaGraphicsUnmapResources(1, &cuda_pbo, 0));

        // 5) Draw to screen
        glClear(GL_COLOR_BUFFER_BIT);

        glRasterPos2i(0, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        glfwSwapBuffers(win);
    }

    // Cleanup
    CHECK(cudaGraphicsUnregisterResource(cuda_pbo));
    glDeleteBuffers(1, &pbo);

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    free(h_board);

    glfwDestroyWindow(win);
    glfwTerminate();
}