// gol_CPUlogic_grafica.cu
// Game of Life: CPU SIMD logic (SSE) + GPU rendering (CUDA) + OpenGL visualization
// Features: zoom/pan, SIMD optimization, real-time rendering

// COMPILE: nvcc filename.cu src/glad.c -I"include" -L"build" -lglfw3 -lopengl32 -lgdi32 -luser32 -Xcompiler "/EHsc /MD" -o filename.exe

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <immintrin.h>
#include <cstring>

// Aggiungi la libreria time.h per il timing POSIX
#if defined(__APPLE__) || defined(__linux__)
#include <time.h>
#else
// Mantieni windows.h solo per Windows
#include <windows.h>
#endif

// --- Funzioni di Timing Portatili ---

// Struttura e funzione per il timing in ambienti POSIX (Linux/macOS)
#if defined(__APPLE__) || defined(__linux__)
static double get_time_posix() {
    struct timespec t;
    // Usa CLOCK_MONOTONIC per misurare il tempo trascorso
    clock_gettime(CLOCK_MONOTONIC, &t);
    // Ritorna il tempo in secondi
    return (double)t.tv_sec + (double)t.tv_nsec / 1000000000.0;
}
#endif

using u8 = unsigned char;

#define CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ==================== HOST CPU SIMD LOGIC ====================

// --- Memory allocation with alignment ---
char* aligned_malloc_grid(int padded_size) {
    size_t size = (size_t)padded_size * padded_size;
    char* ptr = NULL;

    #if defined(_WIN32)
        ptr = (char*)_aligned_malloc(size, 16);
    #else
        if (posix_memalign((void**)&ptr, 16, size) != 0) {
            ptr = NULL;
        }
    #endif

    if (ptr != NULL) {
        memset(ptr, 0, size);
    }
    return ptr;
}

void aligned_free_grid(char* ptr) {
    #if defined(_WIN32)
        _aligned_free(ptr);
    #else
        free(ptr);
    #endif
}

// --- Initialization ---
void init_random_reproducible(char* grid, int logic_size, int padded_size, unsigned int seed) {
    float probability = 0.5f;
    srand(seed);

    // Initialize only logical zone (1..logic_size) skipping padding
    for (int i = 1; i <= logic_size; ++i) {
        for (int j = 1; j <= logic_size; ++j) {
            float r = (float)(rand()) / (float)(RAND_MAX);
            grid[i * padded_size + j] = (r < probability) ? 1 : 0;
        }
    }
}

void initialize_glider(char* grid, int padded_size) {
    int r = 10, c = 10;
    grid[r * padded_size + c + 1] = 1;
    grid[(r + 1) * padded_size + c + 2] = 1;
    grid[(r + 2) * padded_size + c] = 1;
    grid[(r + 2) * padded_size + c + 1] = 1;
    grid[(r + 2) * padded_size + c + 2] = 1;
}

// --- SIMD SSE Update Function ---
void update_with_sse(char* current_grid, char* next_grid, int logic_size, int padded_size) {
    const __m128i three_vec = _mm_set1_epi8(3);
    const __m128i two_vec = _mm_set1_epi8(2);
    const __m128i one_vec = _mm_set1_epi8(1);

    // Loop through logical rows
    for (int i = 1; i <= logic_size; ++i) {
        char* row_prev = current_grid + (i - 1) * padded_size;
        char* row_curr = current_grid + (i) * padded_size;
        char* row_next = current_grid + (i + 1) * padded_size;

        // Loop through logical columns in blocks of 16
        for (int j = 1; j <= logic_size; j += 16) {
            // Load overlapping data from previous row
            __m128i prev_left  = _mm_loadu_si128((__m128i*)(row_prev + j - 1));
            __m128i prev_mid   = _mm_loadu_si128((__m128i*)(row_prev + j));
            __m128i prev_right = _mm_loadu_si128((__m128i*)(row_prev + j + 1));

            // Load overlapping data from current row
            __m128i cur_left   = _mm_loadu_si128((__m128i*)(row_curr + j - 1));
            __m128i cur_mid    = _mm_loadu_si128((__m128i*)(row_curr + j));
            __m128i cur_right  = _mm_loadu_si128((__m128i*)(row_curr + j + 1));

            // Load overlapping data from next row
            __m128i next_left  = _mm_loadu_si128((__m128i*)(row_next + j - 1));
            __m128i next_mid   = _mm_loadu_si128((__m128i*)(row_next + j));
            __m128i next_right = _mm_loadu_si128((__m128i*)(row_next + j + 1));

            // Sum all 8 neighbors
            __m128i neigh_alive = _mm_add_epi8(prev_left, cur_left);
            neigh_alive = _mm_add_epi8(neigh_alive, next_left);
            neigh_alive = _mm_add_epi8(neigh_alive, prev_mid);
            neigh_alive = _mm_add_epi8(neigh_alive, next_mid);
            neigh_alive = _mm_add_epi8(neigh_alive, prev_right);
            neigh_alive = _mm_add_epi8(neigh_alive, cur_right);
            neigh_alive = _mm_add_epi8(neigh_alive, next_right);

            // Apply rules: Alive if N==3 OR (current_state AND N==2)
            __m128i is_three = _mm_cmpeq_epi8(neigh_alive, three_vec);
            __m128i is_two = _mm_cmpeq_epi8(neigh_alive, two_vec);

            __m128i new_state_mask = _mm_or_si128(
                is_three,
                _mm_and_si128(cur_mid, is_two)
            );

            // Convert to binary state (0 or 1)
            __m128i next_cells = _mm_and_si128(new_state_mask, one_vec);

            // Store result
            char* target_ptr = next_grid + i * padded_size + j;
            _mm_storeu_si128((__m128i*)target_ptr, next_cells);
        }
    }
}

// ==================== GPU RENDERING KERNEL ====================

__global__ void renderKernel(uchar4* pbo, const char* board, int logic_size, int padded_size, int scale) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    int displayWidth = logic_size * scale;
    int displayHeight = logic_size * scale;

    if (px >= displayWidth || py >= displayHeight) return;

    int cell_x = px / scale + 1; // +1 for padding offset
    int cell_y = py / scale + 1; // +1 for padding offset
    int cell_idx = cell_y * padded_size + cell_x;
    unsigned char c = board[cell_idx] ? 255 : 0;

    int pix_idx = py * displayWidth + px;
    pbo[pix_idx] = make_uchar4(c, c, c, 255);
}

// ==================== OpenGL HELPERS ====================

GLuint createShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);
    GLint ok;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(shader, 1024, NULL, log);
        fprintf(stderr, "Shader compile error: %s\n", log);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

// ==================== CAMERA STATE (globals for callbacks) ====================

static float g_zoom = 1.0f;
static float g_pan_x = 0.5f;
static float g_pan_y = 0.5f;
static bool  g_dragging = false;
static double g_last_mouse_x = 0.0, g_last_mouse_y = 0.0;
static int   g_display_w = 0, g_display_h = 0;
static bool g_paused = false;

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    g_display_w = width;
    g_display_h = height;
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
        g_paused = !g_paused;
}

inline void mouse_to_uv(double mx, double my, float& u, float& v) {
    u = float(mx / double(g_display_w));
    v = float(1.0 - (my / double(g_display_h)));
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    if (!g_dragging) {
        g_last_mouse_x = xpos;
        g_last_mouse_y = ypos;
        return;
    }

    double dx = xpos - g_last_mouse_x;
    double dy = ypos - g_last_mouse_y;

    float du = float(dx / double(g_display_w)) / g_zoom;
    float dv = float(-dy / double(g_display_h)) / g_zoom;

    g_pan_x -= du;
    g_pan_y -= dv;

    if (g_pan_x < 0.0f) g_pan_x = 0.0f;
    if (g_pan_x > 1.0f) g_pan_x = 1.0f;
    if (g_pan_y < 0.0f) g_pan_y = 0.0f;
    if (g_pan_y > 1.0f) g_pan_y = 1.0f;

    g_last_mouse_x = xpos;
    g_last_mouse_y = ypos;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            g_dragging = true;
            double mx, my;
            glfwGetCursorPos(window, &mx, &my);
            g_last_mouse_x = mx;
            g_last_mouse_y = my;
        } else if (action == GLFW_RELEASE) {
            g_dragging = false;
        }
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    double mx, my;
    glfwGetCursorPos(window, &mx, &my);
    float before_u, before_v;
    mouse_to_uv(mx, my, before_u, before_v);

    float old_zoom = g_zoom;
    float zoom_step = powf(1.1f, (float)yoffset);
    g_zoom *= zoom_step;
    if (g_zoom < 0.05f) g_zoom = 0.05f;
    if (g_zoom > 64.0f) g_zoom = 64.0f;

    g_pan_x = before_u - (before_u - g_pan_x) * (old_zoom / g_zoom);
    g_pan_y = before_v - (before_v - g_pan_y) * (old_zoom / g_zoom);

    if (g_pan_x < 0.0f) g_pan_x = 0.0f;
    if (g_pan_x > 1.0f) g_pan_x = 1.0f;
    if (g_pan_y < 0.0f) g_pan_y = 0.0f;
    if (g_pan_y > 1.0f) g_pan_y = 1.0f;
}

// ==================== MAIN ====================

int main(int argc, char** argv) {
    const int LOGIC_SIZE = 2048;  // Grid size (can be adjusted)
    const int PADDED_SIZE = LOGIC_SIZE + 2;  // Padding for SIMD
    const int SCALE = 4;  // pixels per cell
    const int displayWidth = LOGIC_SIZE * SCALE;
    const int displayHeight = LOGIC_SIZE * SCALE;
    const int WINDOW_WIDTH = 1280;
    const int WINDOW_HEIGHT = 720;
    
    int steps = 0;             // contatore step eseguiti

    g_display_w = displayWidth;
    g_display_h = displayHeight;

    // Variabili per tempo
    double time_start, time_end, time_tot;
    double time_start_step, time_end_step, time_step; 
    // Variabili Windows per i contatori ad alta risoluzione
    #if defined(_WIN32)
        LARGE_INTEGER frequency_win;
        QueryPerformanceFrequency(&frequency_win);
    #endif

    // ========== CPU HOST MEMORY ==========
    char* h_grid_a = aligned_malloc_grid(PADDED_SIZE);
    char* h_grid_b = aligned_malloc_grid(PADDED_SIZE);

    if (!h_grid_a || !h_grid_b) {
        fprintf(stderr, "CPU memory allocation failed\n");
        return -1;
    }

    // Initialize grid
    init_random_reproducible(h_grid_a, LOGIC_SIZE, PADDED_SIZE, 42);
    //initialize_glider(h_grid_a, PADDED_SIZE);

    char* h_current = h_grid_a;
    char* h_next = h_grid_b;

    // ========== GPU MEMORY FOR RENDERING ==========
    size_t pbo_bytes = (size_t)displayWidth * displayHeight * sizeof(uchar4);
    
    // Init GLFW
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    GLFWwindow* win = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, 
                                       "CPU SIMD GoL + GPU Render - Zoom & Pan", NULL, NULL);
    if (!win) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(win);
    glfwSetFramebufferSizeCallback(win, framebuffer_size_callback);
    glfwSetKeyCallback(win, key_callback);
    glfwSetCursorPosCallback(win, cursor_position_callback);
    glfwSetMouseButtonCallback(win, mouse_button_callback);
    glfwSetScrollCallback(win, scroll_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        fprintf(stderr, "Failed to initialize GLAD\n");
        glfwDestroyWindow(win);
        glfwTerminate();
        return -1;
    }

    int fbW, fbH;
    glfwGetFramebufferSize(win, &fbW, &fbH);
    glViewport(0, 0, fbW, fbH);

    // ========== PBO & CUDA INTEROP ==========
    GLuint pbo = 0;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, (GLsizeiptr)pbo_bytes, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaGraphicsResource* cuda_pbo = nullptr;
    CHECK(cudaGraphicsGLRegisterBuffer(&cuda_pbo, pbo, cudaGraphicsMapFlagsWriteDiscard));

    // ========== GPU DEVICE MEMORY FOR BOARD (for rendering only) ==========
    char *d_board = nullptr;
    size_t board_gpu_bytes = (size_t)PADDED_SIZE * PADDED_SIZE * sizeof(char);
    CHECK(cudaMalloc(&d_board, board_gpu_bytes));

    // ========== CUDA KERNEL CONFIG ==========
    const int BLOCK_RENDER_X = 16;
    const int BLOCK_RENDER_Y = 16;
    dim3 dimBlockRender(BLOCK_RENDER_X, BLOCK_RENDER_Y);
    dim3 dimGridRender(
        (displayWidth + dimBlockRender.x - 1) / dimBlockRender.x,
        (displayHeight + dimBlockRender.y - 1) / dimBlockRender.y
    );

    // ========== SHADERS & OPENGL SETUP ==========
    const char* vs_src = R"(
        #version 330 core
        layout(location = 0) in vec2 aPos;
        layout(location = 1) in vec2 aTex;
        out vec2 TexCoord;
        void main() {
            TexCoord = aTex;
            gl_Position = vec4(aPos, 0.0, 1.0);
        }
    )";

    const char* fs_src = R"(
        #version 330 core
        in vec2 TexCoord;
        out vec4 FragColor;
        uniform sampler2D screenTex;
        uniform vec2 u_pan;
        uniform float u_zoom;
        void main() {
            vec2 centered = TexCoord - vec2(0.5);
            centered /= u_zoom;
            vec2 uv = centered + u_pan;
            if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
                FragColor = vec4(0.0, 0.0, 0.0, 1.0);
            } else {
                FragColor = texture(screenTex, uv);
            }
        }
    )";

    GLuint vs = createShader(GL_VERTEX_SHADER, vs_src);
    GLuint fs = createShader(GL_FRAGMENT_SHADER, fs_src);
    if (!vs || !fs) {
        fprintf(stderr, "Shader compilation failed\n");
        return -1;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    {
        GLint ok;
        glGetProgramiv(program, GL_LINK_STATUS, &ok);
        if (!ok) {
            char log[1024];
            glGetProgramInfoLog(program, 1024, NULL, log);
            fprintf(stderr, "Shader link error: %s\n", log);
            return -1;
        }
    }
    glDeleteShader(vs);
    glDeleteShader(fs);

    // Quad
    float quadVertices[] = {
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f,
        -1.0f,  1.0f,  0.0f, 1.0f
    };

    GLuint quadVAO = 0, quadVBO = 0;
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);

    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Texture
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, displayWidth, displayHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    glUseProgram(program);
    glUniform1i(glGetUniformLocation(program, "screenTex"), 0);

    GLint loc_pan = glGetUniformLocation(program, "u_pan");
    GLint loc_zoom = glGetUniformLocation(program, "u_zoom");

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    printf("CPU SIMD Game of Life + GPU Rendering\n");
    printf("Grid size: %dx%d\n", LOGIC_SIZE, LOGIC_SIZE);
    printf("Controls: LMB drag=pan, scroll=zoom, SPACE=pause, ESC=exit\n");

    // ========== MAIN LOOP ==========
    // ISTANTE DI INIZIO MAIN LOOP
    #if defined(__APPLE__) || defined(__linux__)
        time_start = get_time_posix();
    #else // Windows
        LARGE_INTEGER start_win_tot;
        QueryPerformanceCounter(&start_win_tot);
        time_start = (double)start_win_tot.QuadPart;
    #endif

    while (!glfwWindowShouldClose(win) && steps < 1000) {
        glfwPollEvents();

        // ISTANTE DI INIZIO SINGOLO STEP
        #if defined(__APPLE__) || defined(__linux__)
            time_start_step = get_time_posix();
        #else // Windows
            LARGE_INTEGER start_win_step;
            QueryPerformanceCounter(&start_win_step);
            time_start_step = (double)start_win_step.QuadPart;
        #endif

        // 1) CPU SIMD Logic Update
        if (!g_paused) {
            update_with_sse(h_current, h_next, LOGIC_SIZE, PADDED_SIZE);

            // Swap buffers
            char* tmp = h_current;
            h_current = h_next;
            h_next = tmp;

            steps++;
        }

        // 2) Copy CPU grid to GPU for rendering
        CHECK(cudaMemcpy(d_board, h_current, board_gpu_bytes, cudaMemcpyHostToDevice));

        // 3) Map PBO
        CHECK(cudaGraphicsMapResources(1, &cuda_pbo, 0));
        uchar4* pbo_dev_ptr = nullptr;
        size_t mapped_size = 0;
        CHECK(cudaGraphicsResourceGetMappedPointer((void**)&pbo_dev_ptr, &mapped_size, cuda_pbo));

        // 4) Render to PBO
        renderKernel<<<dimGridRender, dimBlockRender>>>(pbo_dev_ptr, d_board, LOGIC_SIZE, PADDED_SIZE, SCALE);
        cudaError_t kerr = cudaGetLastError();
        if (kerr != cudaSuccess) {
            fprintf(stderr, "CUDA render kernel error: %s\n", cudaGetErrorString(kerr));
            cudaGraphicsUnmapResources(1, &cuda_pbo, 0);
            break;
        }
        CHECK(cudaDeviceSynchronize());

        // ISTANTE FINALE SINGOLO STEP
        #if defined(__APPLE__) || defined(__linux__)
            time_end_step = get_time_posix();
            time_step = time_end_step - time_start_step;
        #else // Windows
            LARGE_INTEGER end_win_step;
            QueryPerformanceCounter(&end_win_step);
            time_end_step = (double)end_win_step.QuadPart;
            time_step = (time_end_step - time_start_step) / frequency_win.QuadPart;
        #endif

        // 5) Unmap PBO
        CHECK(cudaGraphicsUnmapResources(1, &cuda_pbo, 0));

        // 6) Upload PBO to Texture
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glBindTexture(GL_TEXTURE_2D, tex);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, displayWidth, displayHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, 0);

        // 7) Draw fullscreen quad with zoom/pan
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(program);
        glUniform2f(loc_pan, g_pan_x, g_pan_y);
        glUniform1f(loc_zoom, g_zoom);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);
        glBindTexture(GL_TEXTURE_2D, 0);

        glfwSwapBuffers(win);
    }
    // ISTANTE FINALE MAIN LOOP
    #if defined(__APPLE__) || defined(__linux__)
        time_end = get_time_posix();
        time_tot = time_end - time_start;
    #else // Windows
        LARGE_INTEGER end_win_tot;
        QueryPerformanceCounter(&end_win_tot);
        time_end = (double)end_win_tot.QuadPart;
        time_tot = (time_end - time_start) / frequency_win.QuadPart;
    #endif

    // ========== CLEANUP ==========
    CHECK(cudaGraphicsUnregisterResource(cuda_pbo));
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
    glDeleteVertexArrays(1, &quadVAO);
    glDeleteBuffers(1, &quadVBO);
    glDeleteProgram(program);

    CHECK(cudaFree(d_board));
    aligned_free_grid(h_grid_a);
    aligned_free_grid(h_grid_b);

    glfwDestroyWindow(win);
    glfwTerminate();

    printf("Simulation completed. Steps executed: %d\n", steps);
    printf("Tempo di esecuzione main loop: %f ms\n", time_tot * 1000);
    printf("Tempo di esecuzione singolo step (ultimo): %f ms\n", time_step * 1000);

    return 0;
}