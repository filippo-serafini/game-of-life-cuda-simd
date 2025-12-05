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

            // Controllo che il vicino non sia fuori dalla griglia di gioco
            // Se è fuori ZERO PADDING => Conto il suo contributo come zero
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

// -------------------- KERNEL 2d Grid/Block con Shared Memory --------------------
// Necessita di una corretta allocazione della shared memory dinamica nel launch (terzo parametro)
__global__ void gol_step_2d2d_shm(u8* src, u8* dst, int width, int height, int RADIUS) {
    
    // Assumiamo che RADIUS sia 1 per il classico Game of Life (vicini 3x3)
    // Sebbene il kernel sia generico per RADIUS, l'ottimizzazione 
    // della shared memory è spesso più efficiente per RADIUS piccoli e fissi.
    if (RADIUS != 1) return; 

    // I blocchi devono essere quadrati (BLOCK_SIZE_X == BLOCK_SIZE_Y) per semplicità
    const int BLOCK_SIZE = blockDim.x; 

    // Dimensioni della Shared Memory: BLOCK_SIZE + 2 * RADIUS
    // Per RADIUS=1, è (BLOCK_SIZE + 2) x (BLOCK_SIZE + 2)
    const int SHM_SIZE = BLOCK_SIZE + 2 * RADIUS;

    // Dichiarazione dinamica della Shared Memory (viene allocata dal launch)
    // Rappresenta il tile del blocco di thread + l'halo (padding)
    extern __shared__ u8 tile_shm[]; 
    u8* shm = (u8*)tile_shm; // Trattiamo il buffer 1D come una griglia 2D in termini di indici

    // Coordinate Globali della Cella che il thread sta calcolando
    int cell_index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int cell_index_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Coordinate Locali della Cella all'interno del blocco
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;

    // ********** Fase 1: Caricamento Dati dalla Global alla Shared Memory **********
    
    // Per un RADIUS=1, ci sono 4 compiti di caricamento:
    // 1. Caricamento del blocco centrale (Core Tile)
    // 2. Caricamento del bordo superiore
    // 3. Caricamento del bordo inferiore
    // 4. Caricamento del bordo sinistro
    // 5. Caricamento del bordo destro
    // 6. Caricamento degli angoli (spesso coperto dai caricamenti di bordo)

    // Un approccio più efficiente e completo (anche per RADIUS > 1) è 
    // far caricare ad ogni thread un elemento della shared memory.
    
    // I thread con indici locali da (0,0) a (BLOCK_SIZE-1, BLOCK_SIZE-1)
    // caricano la parte CORE (interna) della shared memory, corrispondente
    // alle celle che calcoleranno.
    
    // Indice 1D nella Global Memory per la cella Core caricata da questo thread
    int global_core_x = blockIdx.x * BLOCK_SIZE + local_x;
    int global_core_y = blockIdx.y * BLOCK_SIZE + local_y;
    int global_core_mem_idx = global_core_y * width + global_core_x;

    // Indice 1D nella Shared Memory per la cella Core caricata da questo thread
    int shm_core_x = local_x + RADIUS; // Shm Index (con padding)
    int shm_core_y = local_y + RADIUS;
    int shm_core_mem_idx = shm_core_y * SHM_SIZE + shm_core_x;
    
    // I thread che sono "in bounds" nella Global Memory caricano il valore
    if (global_core_x < width && global_core_y < height) {
        shm[shm_core_mem_idx] = src[global_core_mem_idx];
    } else {
        // Se fuori dai bordi della griglia, applica ZERO PADDING
        shm[shm_core_mem_idx] = 0; 
    }

    // Caricamento dell'HALO (Padding): Vengono caricate le celle adiacenti
    // L'halo deve essere caricato da thread che "esistono" all'interno del blocco.
    
    // I thread in (0, y), (BLOCK_SIZE-1, y), (x, 0), (x, BLOCK_SIZE-1)
    // possono essere riutilizzati per caricare l'halo.
    // L'approccio più semplice è usare i thread che caricano i bordi del Core 
    // per caricare anche l'Halo adiacente.

    // Caricamento del bordo superiore e inferiore (Halo Verticale)
    if (local_y < RADIUS) { // Thread per il bordo superiore del blocco (local_y=0)
        // Carica la riga superiore di halo (shm_y = 0)
        int global_halo_y_top = blockIdx.y * BLOCK_SIZE - 1 + local_y; 
        int global_halo_x = global_core_x; // Stessa X del Core
        
        int shm_halo_y_top = local_y; // y=0 (Halo Top)
        int shm_halo_x = shm_core_x;
        int shm_halo_mem_idx = shm_halo_y_top * SHM_SIZE + shm_halo_x;
        
        if (global_halo_x < width && global_halo_y_top >= 0) {
            shm[shm_halo_mem_idx] = src[global_halo_y_top * width + global_halo_x];
        } else {
            shm[shm_halo_mem_idx] = 0; // Zero Padding
        }
    }
    
    if (local_y >= BLOCK_SIZE - RADIUS) { // Thread per il bordo inferiore del blocco (local_y=BLOCK_SIZE-1)
        // Carica la riga inferiore di halo (shm_y = BLOCK_SIZE + 1)
        int global_halo_y_bottom = blockIdx.y * BLOCK_SIZE + BLOCK_SIZE + (local_y - (BLOCK_SIZE - 1));
        int global_halo_x = global_core_x;
        
        int shm_halo_y_bottom = BLOCK_SIZE + 1 + (local_y - (BLOCK_SIZE - 1)); // y=BLOCK_SIZE+1 (Halo Bottom)
        int shm_halo_x = shm_core_x;
        int shm_halo_mem_idx = shm_halo_y_bottom * SHM_SIZE + shm_halo_x;
        
        if (global_halo_x < width && global_halo_y_bottom < height) {
            shm[shm_halo_mem_idx] = src[global_halo_y_bottom * width + global_halo_x];
        } else {
            shm[shm_halo_mem_idx] = 0; // Zero Padding
        }
    }

    // Caricamento del bordo sinistro e destro (Halo Orizzontale)
    if (local_x < RADIUS) { // Thread per il bordo sinistro del blocco (local_x=0)
        // Carica la colonna sinistra di halo (shm_x = 0)
        int global_halo_x_left = blockIdx.x * BLOCK_SIZE - 1 + local_x;
        int global_halo_y = global_core_y; // Stessa Y del Core
        
        int shm_halo_x_left = local_x; // x=0 (Halo Left)
        int shm_halo_y = shm_core_y;
        int shm_halo_mem_idx = shm_halo_y * SHM_SIZE + shm_halo_x_left;
        
        if (global_halo_x_left >= 0 && global_halo_y < height) {
            shm[shm_halo_mem_idx] = src[global_halo_y * width + global_halo_x_left];
        } else {
            shm[shm_halo_mem_idx] = 0; // Zero Padding
        }
    }

    if (local_x >= BLOCK_SIZE - RADIUS) { // Thread per il bordo destro del blocco (local_x=BLOCK_SIZE-1)
        // Carica la colonna destra di halo (shm_x = BLOCK_SIZE + 1)
        int global_halo_x_right = blockIdx.x * BLOCK_SIZE + BLOCK_SIZE + (local_x - (BLOCK_SIZE - 1));
        int global_halo_y = global_core_y;
        
        int shm_halo_x_right = BLOCK_SIZE + 1 + (local_x - (BLOCK_SIZE - 1)); // x=BLOCK_SIZE+1 (Halo Right)
        int shm_halo_y = shm_core_y;
        int shm_halo_mem_idx = shm_halo_y * SHM_SIZE + shm_halo_x_right;
        
        if (global_halo_x_right < width && global_halo_y < height) {
            shm[shm_halo_mem_idx] = src[global_halo_y * width + global_halo_x_right];
        } else {
            shm[shm_halo_mem_idx] = 0; // Zero Padding
        }
    }
    
    // Nota sugli angoli: Il caricamento degli angoli è più complesso, 
    // ma spesso viene gestito implicitamente dai thread di bordo o 
    // con un caricamento esplicito se non coperto. 
    // Per semplicità e considerando RADIUS=1, ci concentriamo sui bordi principali.

    // Sincronizzazione per assicurarsi che tutti i dati del tile + halo siano in shared memory
    __syncthreads(); 
    
    // Controlla che la cella Core sia all'interno dei limiti della griglia originale
    if (cell_index_x >= width || cell_index_y >= height) {
        return; 
    }

    // ********** Fase 2: Calcolo (Utilizzando la Shared Memory) **********

    int neighbors_alive = 0;
    
    // I thread devono ora accedere alla shared memory a partire dalle loro coordinate locali (con padding)
    int shm_center_x = local_x + RADIUS;
    int shm_center_y = local_y + RADIUS;
    
    // Iterazione sui vicini (ora dentro la shared memory)
    for (int dy = -RADIUS; dy <= RADIUS; ++dy) 
    {
        for (int dx = -RADIUS; dx <= RADIUS; ++dx) 
        {
            int neighbor_shm_x = shm_center_x + dx;
            int neighbor_shm_y = shm_center_y + dy;

            // L'indice 1D nella Shared Memory
            int neighbor_shm_idx = neighbor_shm_y * SHM_SIZE + neighbor_shm_x;

            // Non è necessario un controllo sui limiti (neighbor_shm_x/y)
            // perché l'iterazione si muove sempre all'interno della matrice (SHM_SIZE x SHM_SIZE)
            // che include l'halo (che è stato caricato con Zero Padding se fuori dai bordi globali).
            neighbors_alive += shm[neighbor_shm_idx];
        }
    }
    
    // Il valore della cella centrale (dalla shared memory)
    u8 cell_value = shm[shm_center_y * SHM_SIZE + shm_center_x]; 
    neighbors_alive -= cell_value; // Escludo la cella centrale dal conteggio
    u8 cell_result = 0;
    
    // Applicazione delle regole del Game of Life
    cell_result = (neighbors_alive == 3) || (cell_value && (neighbors_alive == 2));

    // ********** Fase 3: Scrittura del Risultato nella Global Memory **********
    int cell_mem_idx = cell_index_y * width + cell_index_x;
    dst[cell_mem_idx] = cell_result;
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
    int radius = 1;

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

    // --- CALCOLO DIMENSIONE SHARED MEMORY ---
    // SHM_SIZE = BLOCK_SIZE + 2 * RADIUS
    const int SHM_SIZE = BLOCK_SIZE_X + 2 * radius; // Assumendo BLOCK_SIZE_X == BLOCK_SIZE_Y
    // Dimensione totale in byte: SHM_SIZE * SHM_SIZE * sizeof(u8)
    size_t shm_bytes = SHM_SIZE * SHM_SIZE * sizeof(u8);

    u8* src = d_a;
    u8* dst = d_b;

    // Kernel execution
    for (int s = 0; s < steps; ++s) {
        //modificare per scegliere il kernel
        //gol_step_2d2d<<<dimGrid, dimBlock>>>(src, dst, width, height, radius);
        gol_step_2d2d_shm<<<dimGrid, dimBlock, shm_bytes>>>(src, dst, width, height, radius);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        // traferisco la griglia GPU -> CPU
        CHECK(cudaMemcpy(h_board, dst, total_bytes, cudaMemcpyDeviceToHost)); 

        // Swap buffers
        u8* tmp = src;
        src = dst;
        dst = tmp;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    free(h_board);
}