// gol_zoom_pan_texture.cu
// CUDA Game of Life: 2D grid/block + OpenGL PBO interop
// Rendering via PBO -> Texture -> Fullscreen Quad (OpenGL Core)
// Continuous zoom (MMB) and pan (LMB)

//COMANDO PER COMPILARE grafica avanzata
//nvcc filename.cu src/glad.c -I"include" -L"build" -lglfw3 -lopengl32 -lgdi32 -luser32 -Xcompiler "/EHsc /MD" -o filename.exe

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>

using u8 = unsigned char;

#define CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// -------------------- GAME OF LIFE KERNEL 2D --------------------
__global__ void gol_step_2d2d(u8* src, u8* dst, int width, int height, int RADIUS) {
    int cell_index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int cell_index_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (cell_index_x >= width || cell_index_y >= height) return;

    int cell_mem_idx = cell_index_y * width + cell_index_x;

    int neighbors_alive = 0;
    for (int dy = -RADIUS; dy <= RADIUS; ++dy) {
        for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
            int neighbor_cell_x = cell_index_x + dx;
            int neighbor_cell_y = cell_index_y + dy;

            if(neighbor_cell_x >= 0 && neighbor_cell_x < width 
                && neighbor_cell_y >= 0 && neighbor_cell_y < height)
                {
                    neighbors_alive += src[neighbor_cell_y * width + neighbor_cell_x];
                }
        }
    }

    u8 cell_value = src[cell_mem_idx];
    neighbors_alive -= cell_value;
    u8 cell_result = (neighbors_alive == 3) || (cell_value && (neighbors_alive == 2));
    dst[cell_mem_idx] = cell_result;
}

// -------------------- RENDER KERNEL (writes RGBA to PBO) --------------------
__global__ void renderKernel(uchar4* pbo, const u8* board, int width, int height, int scale) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    int displayWidth = width * scale;
    int displayHeight = height * scale;

    if (px >= displayWidth || py >= displayHeight) return;

    int cell_x = px / scale;
    int cell_y = py / scale;
    int cell_idx = cell_y * width + cell_x;
    unsigned char c = board[cell_idx] ? 255 : 0;

    int pix_idx = py * displayWidth + px;
    pbo[pix_idx] = make_uchar4(c, c, c, 255);
}

// -------------------- Helpers host --------------------
void random_board(u8* board, int width, int height, float alive_prob = 0.2f) {
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            board[y * width + x] = (float(rand()) / RAND_MAX) < alive_prob ? 1 : 0;
}

void initialize_glider(u8* board, int width, int height) {
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            board[y * width + x] = 0;

    int r = 10, c = 10;
    if (r + 2 < height && c + 2 < width) {
        board[r * width + c + 1]         = 1;
        board[(r + 1) * width + c + 2]   = 1;
        board[(r + 2) * width + c]       = 1;
        board[(r + 2) * width + c + 1]   = 1;
        board[(r + 2) * width + c + 2]   = 1;
    }
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

// -------------------- Shader helper --------------------
GLuint createShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);
    GLint ok;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(shader, 1024, NULL, log);
        fprintf(stderr, "Shader compile error: %s", log);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

// -------------------- Camera (zoom/pan) state - globals for callbacks --------------------
static float g_zoom = 1.0f;            // zoom factor (1.0 = default)
static float g_pan_x = 0.5f;           // pan center in texture space [0..1]
static float g_pan_y = 0.5f;
static bool  g_dragging = false;
static double g_last_mouse_x = 0.0, g_last_mouse_y = 0.0;
static int   g_display_w = 0, g_display_h = 0;
static bool g_paused = false;          // pausa simulazione

// Forward declarations for callbacks
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    // aggiorna viewport; la texture/PBO non vengono ricreati 
    // (finestra non ridimensionabile nella build corrente)
    glViewport(0, 0, width, height);
    g_display_w = width; g_display_h = height;
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) glfwSetWindowShouldClose(window, GLFW_TRUE);

    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
        g_paused = !g_paused;
    }
}

// Convert mouse pixel coords to texture UV [0..1]
inline void mouse_to_uv(double mx, double my, float& u, float& v) {
    // glfw gives cursor pos relative to top-left? Actually it's top-left origin for window coords,
    // but our texture sampling uses TexCoord with v=0 at bottom; we assume mouse y=0 at top.
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

    // delta in texture coordinates, account for zoom
    float du = float(dx / double(g_display_w)) / g_zoom;
    float dv = float(-dy / double(g_display_h)) / g_zoom; // invert Y since mouse Y grows down

    g_pan_x -= du; // subtract because moving mouse right should pan view right -> decrease center u
    g_pan_y -= dv;

    // clamp pan to [0,1]
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
            double mx, my; glfwGetCursorPos(window, &mx, &my);
            g_last_mouse_x = mx; g_last_mouse_y = my;
        } else if (action == GLFW_RELEASE) {
            g_dragging = false;
        }
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    // Zoom toward mouse position
    double mx, my; glfwGetCursorPos(window, &mx, &my);
    float before_u, before_v; mouse_to_uv(mx, my, before_u, before_v);

    // update zoom (smooth exponential)
    float old_zoom = g_zoom;
    float zoom_step = powf(1.1f, (float)yoffset); // yoffset positive -> zoom in
    g_zoom *= zoom_step;
    if (g_zoom < 0.05f) g_zoom = 0.05f;
    if (g_zoom > 64.0f) g_zoom = 64.0f;

    // Adjust pan so that the point under the cursor stays under the cursor
    // new_pan + (before - new_pan) = before => compute new_pan = before - (before - old_pan) * (old_zoom/new_zoom)
    g_pan_x = before_u - (before_u - g_pan_x) * (old_zoom / g_zoom);
    g_pan_y = before_v - (before_v - g_pan_y) * (old_zoom / g_zoom);

    // clamp
    if (g_pan_x < 0.0f) g_pan_x = 0.0f;
    if (g_pan_x > 1.0f) g_pan_x = 1.0f;
    if (g_pan_y < 0.0f) g_pan_y = 0.0f;
    if (g_pan_y > 1.0f) g_pan_y = 1.0f;
}

// -------------------- MAIN --------------------
int main(int argc, char** argv) {
    // Parametri logici
    const int width = 2048;
    const int height = 2048;
    const int SCALE = 4; // pixel per cell (display size = width*SCALE)
    const int radius = 1;

    int steps = 0;

    const int displayWidth = width * SCALE;
    const int displayHeight = height * SCALE;
    
    // Window size
    const int WINDOW_WIDTH  = 1280;
    const int WINDOW_HEIGHT = 720;

    g_display_w = displayWidth; g_display_h = displayHeight;

    size_t cells = size_t(width) * size_t(height);
    size_t board_bytes = cells * sizeof(u8);
    size_t display_pixels = size_t(displayWidth) * size_t(displayHeight);
    size_t pbo_bytes = display_pixels * sizeof(uchar4);

    // Init GLFW
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW");
        return -1;
    }

    // Request OpenGL Core 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    // disabilita resizing per semplicità
    // altrimenti bisognerebbe ricreare PBO/texture nel callback
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    GLFWwindow* win = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "CUDA Game of Life - Zoom & Pan", NULL, NULL);
    if (!win) {
        fprintf(stderr, "Failed to create GLFW window");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(win);
    glfwSetFramebufferSizeCallback(win, framebuffer_size_callback);
    glfwSetKeyCallback(win, key_callback);
    glfwSetCursorPosCallback(win, cursor_position_callback);
    glfwSetMouseButtonCallback(win, mouse_button_callback);
    glfwSetScrollCallback(win, scroll_callback);

    // Load GL
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        fprintf(stderr, "Failed to initialize GLAD");
        glfwDestroyWindow(win);
        glfwTerminate();
        return -1;
    }

    // Viewport setup
    int fbW, fbH;
    glfwGetFramebufferSize(win, &fbW, &fbH);
    glViewport(0, 0, fbW, fbH);

    // -------------------- PBO & CUDA interop --------------------
    GLuint pbo = 0;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, (GLsizeiptr)pbo_bytes, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaGraphicsResource* cuda_pbo = nullptr;
    CHECK(cudaGraphicsGLRegisterBuffer(&cuda_pbo, pbo, cudaGraphicsMapFlagsWriteDiscard));

    // -------------------- Host & Device memory --------------------
    u8* h_board = (u8*)malloc(board_bytes);
    if (!h_board) {
        fprintf(stderr, "Host allocation failed");
        return -1;
    }
    srand((unsigned)time(NULL));
    
    // diverse modalità di inizializzazione della griglia
    //random_board(h_board, width, height, 0.15f);
    //initialize_glider(h_board, width, height);
    init_random_reproducible(h_board, width, height, 42); // seed 42

    u8 *d_a = nullptr, *d_b = nullptr;
    CHECK(cudaMalloc(&d_a, board_bytes));
    CHECK(cudaMalloc(&d_b, board_bytes));
    CHECK(cudaMemcpy(d_a, h_board, board_bytes, cudaMemcpyHostToDevice));

    // -------------------- Kernel config --------------------
    const int BLOCK_GAME_X = 16;
    const int BLOCK_GAME_Y = 16;
    dim3 dimBlockGame(BLOCK_GAME_X, BLOCK_GAME_Y);
    dim3 dimGridGame(
        (width  + dimBlockGame.x - 1) / dimBlockGame.x,
        (height + dimBlockGame.y - 1) / dimBlockGame.y
    );

    const int BLOCK_RENDER_X = 16;
    const int BLOCK_RENDER_Y = 16;
    dim3 dimBlockRender(BLOCK_RENDER_X, BLOCK_RENDER_Y);
    dim3 dimGridRender(
        (displayWidth  + dimBlockRender.x - 1) / dimBlockRender.x,
        (displayHeight + dimBlockRender.y - 1) / dimBlockRender.y
    );

    u8* src = d_a;
    u8* dst = d_b;

    // -------------------- Shader, VAO, VBO, Texture --------------------
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

    // Fragment shader performs zoom/pan in texture space using uniforms
    const char* fs_src = R"(
        #version 330 core
        in vec2 TexCoord;
        out vec4 FragColor;
        uniform sampler2D screenTex;
        uniform vec2 u_pan;   // center in UV [0..1]
        uniform float u_zoom; // zoom factor
        void main() {
            // map TexCoord [0..1] to centered coords [-0.5..0.5]
            vec2 centered = TexCoord - vec2(0.5);
            // apply zoom
            centered /= u_zoom;
            // map back and shift by pan (pan is texture center)
            vec2 uv = centered + u_pan;
            // sample (outside -> black)
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
        fprintf(stderr, "Shader compilation failed");
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
            fprintf(stderr, "Shader link error: %s", log);
            return -1;
        }
    }
    glDeleteShader(vs);
    glDeleteShader(fs);

    // Quad (two triangles) NDC + texcoords
    float quadVertices[] = {
        // pos      // tex
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

    // Create texture to hold PBO contents
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

    // -------------------- Main loop --------------------
    while (!glfwWindowShouldClose(win)) { //&& steps < 100 per aggiungere un limite di step
        glfwPollEvents();

        // 1) Step Gol
        if (!g_paused)
        {
            gol_step_2d2d<<<dimGridGame, dimBlockGame>>>(src, dst, width, height, radius);
            cudaError_t kerr = cudaGetLastError();
            if (kerr != cudaSuccess) {
                fprintf(stderr, "CUDA kernel error (gol): %s", cudaGetErrorString(kerr));
                break;
            }
            CHECK(cudaDeviceSynchronize());

            // swap buffers
            u8* tmp = src; 
            src = dst; 
            dst = tmp;

            steps++;
        }

        // 2) Map PBO and get pointer
        CHECK(cudaGraphicsMapResources(1, &cuda_pbo, 0));
        uchar4* pbo_dev_ptr = nullptr;
        size_t mapped_size = 0;
        CHECK(cudaGraphicsResourceGetMappedPointer((void**)&pbo_dev_ptr, &mapped_size, cuda_pbo));

        // 3) Render into PBO
        renderKernel<<<dimGridRender, dimBlockRender>>>(pbo_dev_ptr, src, width, height, SCALE);
        cudaError_t kerr = cudaGetLastError();
        if (kerr != cudaSuccess) {
            fprintf(stderr, "CUDA kernel error (render): %s", cudaGetErrorString(kerr));
            cudaGraphicsUnmapResources(1, &cuda_pbo, 0);
            break;
        }
        CHECK(cudaDeviceSynchronize());

        // 4) Unmap PBO
        CHECK(cudaGraphicsUnmapResources(1, &cuda_pbo, 0));

        // 5) Upload PBO -> Texture
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glBindTexture(GL_TEXTURE_2D, tex);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, displayWidth, displayHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, 0);

        // 6) Draw fullscreen quad with pan/zoom
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(program);
        // Set pan and zoom uniforms
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

    // -------------------- Cleanup --------------------
    CHECK(cudaGraphicsUnregisterResource(cuda_pbo));
    glDeleteBuffers(1, &pbo);

    glDeleteTextures(1, &tex);

    glDeleteVertexArrays(1, &quadVAO);
    glDeleteBuffers(1, &quadVBO);
    glDeleteProgram(program);

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    free(h_board);

    glfwDestroyWindow(win);
    glfwTerminate();

    return 0;
}
