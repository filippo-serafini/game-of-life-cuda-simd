/*
*       VERSIONE CUDA GRIGLIA 2D BLOCCO 2D (16x16) E SHARED MEMORY
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

// Definiamo le dimensioni del blocco a compile-time per la Shared Memory
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define RADIUS 1

// Dichiarazioni in constant memory (scope globale)
__constant__ const int dev_width;
__constant__ const int dev_height;

// La tile in Shared Memory deve contenere il blocco + le righe e colonne
// di zero padding o appartenenti ad altri blocchi per le celle sui bordi.
// Dimensione Shared: (16 + 2) x (16 + 2) = 18x18
#define SM_W (BLOCK_DIM_X + 2 * RADIUS)
#define SM_H (BLOCK_DIM_Y + 2 * RADIUS)

__global__ void gol_step_shared(u8* src, u8* dst) {
    
    // Allocazione statica Shared Memory 
    __shared__ u8 tile[SM_H][SM_W];

    // COORDINATE GLOBALI del thread (per leggere da Global Memory)
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    int global_idx = global_y * dev_width + global_x;

    // COORDINATE LOCALI nel BLOCCO (0..15)
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;

    // COORDINATE LOCALI nella SHARED MEMORY
    // Aggiungo RADIUS per considerare in posizione 0 e
    // blockDim.y + 1 le righe/colonne di padding (zeri) o di altri blocchi.
    // riga 0 local <=> riga 1 shared
    int shared_x = local_x + RADIUS;
    int shared_y = local_y + RADIUS;

    // -----------------------------------
    //   CARICAMENTO IN SHARED MEMORY
    // -----------------------------------

    // 1- Carico la cella centrale

    // Controllo per indici oltre la griglia di gioco.
    if (global_x < dev_width && global_y < dev_height) {
        // Accesso alla memoria globale *src*
        tile[shared_y][shared_x] = src[global_idx];
    } else {
        tile[shared_y][shared_x] = 0;
    }
     /*
       Accessi in memoria => ogni thread accede alla sua cella in maniera coalescente. 
       Il thread 0 accede a x, il thread 1 accede a x+1 ecc...
       Essendo 32 thread x warp => ogni warp chiede 32 byte => un'unica transazione
       Il mem controller vede questa cosa e raccoglie tutte le richieste del warp in 
       una sola transazione da 32.
       [Warp divergence] non di impatto => In questo caso stesso numero di celle e thread.
    */

    // 2- Caricamento dei vicini
    /*
        Le celle di gioco dei vicini vengono completamente caricate dagli altri thread del blocco
        solo per le celle dalla riga 1 a riga dimBlock.y-1 e da colonna 1 a colonna dimBlock.x-1
        dall'istruzione precedente (necessaria sincronizzazione tra thread).
        Per le altre celle di bordo, devo accedere alla memoria globale.
    */

    // 2.1- Riga 0 del blocco
    // Devo caricare l'ultima riga del blocco superiore 
    // oppure tutti 0 se è la riga 0 della griglia di gioco totale (zero-padding)
    if (local_y < RADIUS) { 
        int load_y = global_y - RADIUS; // riga precedente
        if (load_y >= 0 && global_x < dev_width)
            tile[shared_y - RADIUS][shared_x] = src[load_y * dev_width + global_x];
        else
            tile[shared_y - RADIUS][shared_x] = 0;
    }
     /*
        Accessi in memoria => solo i thread che accedono alle celle appartenenti alla riga 0 del thread block 
        eseguono questa istruzione ad indirizzi coalesced: thread 0 -> cella[0][0], thread 1 -> cella[0][1]...
        con blocchi 16x16 => 16 thread (metà warp) => 16 byte.
        In questo modo solo 16 dei 32 byte richiesti saranno utilizzati.ù
    */

    // 2.2- Ultima riga del blocco
    // Devo caricare la prima riga del blocco inferiore 
    // oppure tutti 0 se è l'ultima riga della griglia di gioco totale (zero-padding)
    if (local_y >= blockDim.y - RADIUS) {
        int load_y = global_y + RADIUS; // riga successiva
        if (load_y < dev_height && global_x < dev_width)
            tile[shared_y + RADIUS][shared_x] = src[load_y * dev_width + global_x];
        else
            tile[shared_y + RADIUS][shared_x] = 0;
    }
    /*
        Accessi in memoria => come prima
    */

    // 2.3- Colonna 0 del blocco
    // Devo caricare l'ultima colonna del blocco a sinistra 
    // oppure tutti 0 se è la colonna 0 della griglia di gioco totale (zero-padding)
    if (local_x < RADIUS) {
        int load_x = global_x - RADIUS; // colonna precedente
        if (load_x >= 0 && global_y < dev_height)
            tile[shared_y][shared_x - RADIUS] = src[global_y * dev_width + load_x];
        else
            tile[shared_y][shared_x - RADIUS] = 0;
    }
    /*
        Accessi in memoria -> Qui gli accessi sono peggiori e non coalescenti. Ogni warp contiene solo 2 thread 
        che sono associati alla colonna 0 => per accessi row-major mi serve solo un byte del totale
        caricati da una transazione.
    */

    // 2.4- Ultima colonna del blocco
    // Devo caricare la prima colonna del blocco a destra 
    // oppure tutti 0 se è l'ultima colonna della griglia di gioco totale (zero-padding)
    if (local_x >= blockDim.x - RADIUS) {
        int load_x = global_x + RADIUS; // colonna successiva
        if (load_x < dev_width && global_y < dev_height)
            tile[shared_y][shared_x + RADIUS] = src[global_y * dev_width + load_x];
        else
            tile[shared_y][shared_x + RADIUS] = 0;
    }
    /*
        Accessi in memoria -> Come prima
    */

    //2.5- Rimangono solo i vicini NW, NE, SE, SW
    if (local_x < RADIUS && local_y < RADIUS) { // Top-Left
        int load_y = global_y - RADIUS; int load_x = global_x - RADIUS;
        tile[shared_y - RADIUS][shared_x - RADIUS] = (load_x >= 0 && load_y >= 0) ? src[load_y * dev_width + load_x] : 0;
    }
    if (local_x >= blockDim.x - RADIUS && local_y < RADIUS) { // Top-Right
        int load_y = global_y - RADIUS; int load_x = global_x + RADIUS;
        tile[shared_y - RADIUS][shared_x + RADIUS] = (load_x < dev_width && load_y >= 0) ? src[load_y * dev_width + load_x] : 0;
    }
    if (local_x < RADIUS && local_y >= blockDim.y - RADIUS) { // Bottom-Left
        int load_y = global_y + RADIUS; int load_x = global_x - RADIUS;
        tile[shared_y + RADIUS][shared_x - RADIUS] = (load_x >= 0 && load_y < dev_height) ? src[load_y * dev_width + load_x] : 0;
    }
    if (local_x >= blockDim.x - RADIUS && local_y >= blockDim.y - RADIUS) { // Bottom-Right
        int load_y = global_y + RADIUS; int load_x = global_x + RADIUS;
        tile[shared_y + RADIUS][shared_x + RADIUS] = (load_x < dev_width && load_y < dev_height) ? src[load_y * dev_width + load_x] : 0;
    }

    // [Warp Divergence] Purtroppo inevitabile per questo tipo di accesso alle celle di gioco 
    // e uso della shared memory.
    
    // BARRIERA DI SINCRONIZZAZIONE
    // Necessaria per assicurare che tutto il blocco abbia caricato i dati in Shared Mem
    __syncthreads();

    // ---------------------------------------
    //  CALCOLO DEL NUOVO STATO DELLA CELLA
    //  Posso accedere ora solo alla smem
    // ---------------------------------------
    
    // Controllo per indici oltre la griglia di gioco.
    if (global_x >= dev_width || global_y >= dev_height) {
        return;
    }
    // FONDAMENTALE farlo qui se no la barriera __syncthreads non sarà mai
    // raggiunta da tutti i thread se ci sono thread di riempimento del warp => deadlock

    int neighbors_alive = 0;
    for (int dy = -RADIUS; dy <= RADIUS; ++dy) {
        for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
            // Lettura da SMEM
            neighbors_alive += tile[shared_y + dy][shared_x + dx];
        }
    }

    u8 cell_value = tile[shared_y][shared_x];
    // Escludo la cella centrale dal conteggio
    neighbors_alive -= cell_value;

    // Salvataggio del risultato in memoria globale *dst*
    dst[global_idx] = (neighbors_alive == 3) || (cell_value && (neighbors_alive == 2));
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

    // Copia width e height in constant memory
    CHECK(cudaMemcpyToSymbol(dev_width, &width, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(dev_height, &height, sizeof(int)));
 
    // --- DIMENSIONAMENTO DI GRIGLIA E BLOCCHI ---
    dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
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
        gol_step_shared<<<dimGrid, dimBlock>>>(src, dst);
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