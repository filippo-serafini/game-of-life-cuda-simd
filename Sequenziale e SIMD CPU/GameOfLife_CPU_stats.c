#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <immintrin.h> // Include per gli intrinsics SSE
#include <windows.h> // Necessario per le funzioni di performance counter

// --- Costanti ---
#define LOGIC_SIZE 32*5       // Dimensione effettiva della griglia N x N
#define PADDED_SIZE (LOGIC_SIZE + 2) // 66x66 con zero-padding
#define ALIGNMENT 16        // Allineamento richiesto da SSE

// Alloca memoria allineata a 16 byte
char* aligned_malloc_grid() {
    size_t size = PADDED_SIZE * PADDED_SIZE;
    char* ptr = NULL;
    
    #if defined(_WIN32)
        ptr = (char*)_aligned_malloc(size, ALIGNMENT);
    #elif defined(__APPLE__) || defined(__linux__)
        if (posix_memalign((void**)&ptr, ALIGNMENT, size) != 0) {
            ptr = NULL;
        }
    #else
        // Fallback
        ptr = (char*)malloc(size);
    #endif

    if (ptr != NULL) {
        // Zero-inizializza tutto (zero-padding implicito)
        memset(ptr, 0, size);
    }
    return ptr;
}

// Libera memoria allineata
void aligned_free_grid(char* ptr) {
    #if defined(_WIN32)
        _aligned_free(ptr);
    #else
        free(ptr);
    #endif
}

// Inizializza la griglia con un Glider
void initialize_glider(char* grid_data) {
    int r = 10;
    int c = 10;
    
    // Griglia[riga * PADDED_SIZE + colonna]
    grid_data[r * PADDED_SIZE + c + 1] = 1;
    grid_data[(r + 1) * PADDED_SIZE + c + 2] = 1;
    grid_data[(r + 2) * PADDED_SIZE + c] = 1;
    grid_data[(r + 2) * PADDED_SIZE + c + 1] = 1;
    grid_data[(r + 2) * PADDED_SIZE + c + 2] = 1;
    
    return;
}
/*
    Glider:
    . . . . .
    . . # . .
    . . . # .
    . # # # .
    . . . . .
*/

// Stampa la griglia (solo la parte LOGIC_SIZE x LOGIC_SIZE)
void print_grid(const char* grid) {
    printf("----------------------------------------------------------------------------------------------------------------------------------\n");
    for (int i = 1; i <= LOGIC_SIZE; ++i) {
        for (int j = 1; j <= LOGIC_SIZE; ++j) {
            printf("%c ", grid[i * PADDED_SIZE + j] ? '#' : '.');
        }
        printf("\n");
    }
    printf("----------------------------------------------------------------------------------------------------------------------------------\n");
}

// --- Funzione Principale di Calcolo con SSE ---
void update_with_sse(char* current_grid, char* next_grid) {
    // Caricamento costante dei vettori di confronto fuori dal loop
    const __m128i zero_vec = _mm_setzero_si128();
    // Vettori temporanei di utilità per i calcoli
    const __m128i three_vec = _mm_set1_epi8(3); // 0x03 per ogni byte
    const __m128i two_vec = _mm_set1_epi8(2);   // 0x02 per ogni byte    
    const __m128i one_vec = _mm_set1_epi8(1);   // 0x01 per ogni byte
    
    // Loop sulle righe logiche
    for (int i = 1; i <= LOGIC_SIZE; ++i) { 
        // puntatori all'inizio della riga precedente, corrente e successiva
        char* row_prev = current_grid + (i - 1) * PADDED_SIZE; 
        char* row_curr = current_grid + (i) * PADDED_SIZE;
        char* row_next = current_grid + (i + 1) * PADDED_SIZE;
        
        // Loop sulle colonne logiche (j = 1 a 64) in blocchi di 16 celle.
        // Poiché 32 è multiplo di 16, non servono controlli sul bordo destro.
        for (int j = 1; j <= LOGIC_SIZE; j += 16) { 

            // Caricamenti sovrapposti per la riga precedente
            __m128i prev_left  = _mm_loadu_si128((__m128i*)(row_prev + j - 1)); // j-1 .. j+14
            __m128i prev_mid   = _mm_loadu_si128((__m128i*)(row_prev + j));     // j .. j+15
            __m128i prev_right = _mm_loadu_si128((__m128i*)(row_prev + j + 1)); // j+1 .. j+16
            
            // Caricamenti sovrapposti per la riga corrente
            __m128i cur_left  = _mm_loadu_si128((__m128i*)(row_curr + j - 1));
            __m128i cur_mid   = _mm_loadu_si128((__m128i*)(row_curr + j));
            __m128i cur_right = _mm_loadu_si128((__m128i*)(row_curr + j + 1));

            // Caricamenti sovrapposti per la riga successiva
            __m128i next_left  = _mm_loadu_si128((__m128i*)(row_next + j - 1));
            __m128i next_mid   = _mm_loadu_si128((__m128i*)(row_next + j));
            __m128i next_right = _mm_loadu_si128((__m128i*)(row_next + j + 1));


            // Calcolo dei vicini vivi (Somma degli 8 blocchi)
            __m128i neigh_alive = _mm_add_epi8(prev_left, cur_left);
            neigh_alive = _mm_add_epi8(neigh_alive, next_left);
            neigh_alive = _mm_add_epi8(neigh_alive, prev_mid);
            neigh_alive = _mm_add_epi8(neigh_alive, next_mid);
            neigh_alive = _mm_add_epi8(neigh_alive, prev_right);
            neigh_alive = _mm_add_epi8(neigh_alive, cur_right);
            neigh_alive = _mm_add_epi8(neigh_alive, next_right);
            // ^-- Ogni campo del registro avrà il numero corrispondente ai vicini per quella cella:
            // posizione 0 -> numero vicini vivi per cella in posizione 0
            // posizione 1 -> numero vicini vivi per cella in posizione 1...

            // 4. Applicazione delle Regole (Logica Booleana SIMD)
            // Maschere per non introdurre if/else non supportate da paradigma SIMD

            // Maschera N=3
            __m128i is_three = _mm_cmpeq_epi8(neigh_alive, three_vec);
            // Maschera N=2
            __m128i is_two = _mm_cmpeq_epi8(neigh_alive, two_vec);

            // Soppravvivenza => se neighbour_count vale 2 o 3
            // Nascita => se neighbour_count vale 3
            // Morte => se < 2 o > 3
            // Nuovo Stato = (N=3) OR (Stato Attuale AND N=2)
            __m128i new_state_mask = _mm_or_si128(
                is_three,  // Nascita di una cella                                     
                _mm_and_si128(cur_mid, is_two)            
            );
            // ^--- produce 0x00 se false, 0xFF se true

            // Conversione in binario dello stato calcolato
            __m128i next_cells = _mm_and_si128(new_state_mask, one_vec);

            // 5. Memorizzazione del nuovo stato
            char* target_ptr_next = next_grid + i * PADDED_SIZE + j;
            _mm_storeu_si128((__m128i*)target_ptr_next, next_cells);
        }
    }
}

// -------------------------------------------- Versione Sequenziale --------------------------------------------

/* Alloca una matrice (righe x cols) contigua */
static char* allocate_grid() {
    
    char *g = malloc((size_t)LOGIC_SIZE * LOGIC_SIZE * sizeof(char));
    if (!g) {
        fprintf(stderr, "Errore: allocazione memoria fallita\n");
        exit(EXIT_FAILURE);
    }
    return g;
}

// Stampa la griglia (solo la parte LOGIC_SIZE x LOGIC_SIZE)
void print_grid_seq(const char* grid) {
    printf("----------------------------------------------------------------------------------------------------------------------------------\n");
    for (int i = 0; i < LOGIC_SIZE; i++) {
        for (int j = 0; j < LOGIC_SIZE; j++) {
            printf("%c ", grid[i * LOGIC_SIZE + j] ? '#' : '.');
        }
        printf("\n");
    }
    printf("----------------------------------------------------------------------------------------------------------------------------------\n");
}

// Inizializza Glider (Versione Sequenziale)
void initialize_glider_sequential(char* grid_data) {
    memset(grid_data, 0, LOGIC_SIZE * LOGIC_SIZE);
    int r = 10; int c = 10;
    // Indici diretti senza offset padding
    grid_data[r * LOGIC_SIZE + c] = 1;
    grid_data[(r + 1) * LOGIC_SIZE + c + 1] = 1;
    grid_data[(r + 2) * LOGIC_SIZE + c - 1] = 1;
    grid_data[(r + 2) * LOGIC_SIZE + c] = 1;
    grid_data[(r + 2) * LOGIC_SIZE + c + 1] = 1;
}

static void initialize_b(char* grid){
    for(int i = 0; i < LOGIC_SIZE*LOGIC_SIZE; i++)
            grid[i] = 0;
}

static void update_sequential(char* curr_grid, char* next_grid){
    
    for (int row = 1; row <= LOGIC_SIZE; ++row) {
        
        // Calcoliamo l'offset della riga corrente una volta sola per risparmiare moltiplicazioni
        int row_offset = row * PADDED_SIZE;
        
        for (int col = 1; col <= LOGIC_SIZE; ++col) {
            
            int idx = row_offset + col; // Indice della cella corrente
            
            // Conta i vicini sommando direttamente gli offset fissi
            int n = 
                curr_grid[idx - PADDED_SIZE - 1] + // Alto-Sinistra
                curr_grid[idx - PADDED_SIZE]     + // Alto-Centro
                curr_grid[idx - PADDED_SIZE + 1] + // Alto-Destra
                curr_grid[idx - 1]               + // Sinistra
                curr_grid[idx + 1]               + // Destra
                curr_grid[idx + PADDED_SIZE - 1] + // Basso-Sinistra
                curr_grid[idx + PADDED_SIZE]     + // Basso-Centro
                curr_grid[idx + PADDED_SIZE + 1];  // Basso-Destra

            // Applica le regole:
            // Cella viva (1): sopravvive se n == 2 o n == 3
            // Cella morta (0): nasce se n == 3
            // Possiamo compattare la logica:
            
            char is_alive = curr_grid[idx];
            
            // Logica senza branch
            next_grid[idx] = (n == 3) | (is_alive & (n == 2));
        }
    }
}
        
// --- Main Program ---
int main() {
    printf("Game of Life (32x32) C con SSE SIMD\n");

    // Istanti di inizio e fine per l'esecuzione
    // Variabili per il tempo sequenziale
    uint64_t clock_counter_sequential_start, clock_counter_sequential_end;
    LARGE_INTEGER frequency_seq;
    LARGE_INTEGER start_seq, end_seq;
    double time_seq;
    // Variabili per il tempo SIMD
    uint64_t clock_counter_SIMD_start, clock_counter_SIMD_end;
    LARGE_INTEGER frequency_simd;
    LARGE_INTEGER start_simd, end_simd;
    double time_simd;
    
    
    int generations = 1500;

    // --------------------------------------- SEQUENZIALE ---------------------------------------
    
    // Alloca e inizializza le due griglie
    char* grid_a = aligned_malloc_grid();
    char* grid_b = aligned_malloc_grid();

    initialize_glider(grid_a);
    //initialize_b(grid_b);

    char* current = grid_a;
    char* next = grid_b;

    // ISTANTE DI INIZIO
    clock_counter_sequential_start = __rdtsc();
    // Ottiene la frequenza (conta il numero di tick al secondo)
    QueryPerformanceFrequency(&frequency_seq); 
    // Ottiene il valore iniziale del contatore
    QueryPerformanceCounter(&start_seq);

    for (int g = 0; g < generations; ++g) {
        //printf("\nGenerazione %d:\n", g);
        //print_grid_seq(current);
        
        // Calcola la prossima generazione
        update_sequential(current, next);
        
        // Scambia le griglie (doppio buffering)
        char* temp = current;
        current = next;
        next = temp;
    }
    //printf("\nGenerazione %d (Finale):\n", generations);
    //print_grid_seq(current);
    // ISTANTE FINALE
    clock_counter_sequential_end = __rdtsc();
    // Ottiene il valore finale del contatore
    QueryPerformanceCounter(&end_seq);
    // Calcola il tempo in secondi
    time_seq = (double)(end_seq.QuadPart - start_seq.QuadPart) / frequency_seq.QuadPart;    

    aligned_free_grid(grid_a);
    aligned_free_grid(grid_b);

    // ------------------------ SIMD ------------------------

    // Alloca e inizializza le due griglie
    grid_a = aligned_malloc_grid();
    grid_b = aligned_malloc_grid();

    if (!grid_a || !grid_b) {
        fprintf(stderr, "Errore nell'allocazione della memoria allineata.\n");
        return 1;
    }

    initialize_glider(grid_a);

    // Loop principale
    current = grid_a;
    next = grid_b;

    // ISTANTE DI INIZIO
    clock_counter_SIMD_start = __rdtsc();
    // Ottiene la frequenza
    QueryPerformanceFrequency(&frequency_simd);
    // Ottiene il valore iniziale del contatore
    QueryPerformanceCounter(&start_simd);

    for (int g = 0; g < generations; ++g) {
        //printf("\nGenerazione %d:\n", g);
        //print_grid(current);
        
        // Calcola la prossima generazione
        update_with_sse(current, next);
        
        // Scambia le griglie (doppio buffering)
        char* temp = current;
        current = next;
        next = temp;
    }
    
    //printf("\nGenerazione %d (Finale):\n", generations);
    //print_grid(current);

    // ISTANTE FINALE
    clock_counter_SIMD_end = __rdtsc();
    // Ottiene il valore finale del contatore
    QueryPerformanceCounter(&end_simd);
    // Calcola il tempo in secondi
    time_simd = (double)(end_simd.QuadPart - start_simd.QuadPart) / frequency_simd.QuadPart;

    double speedup_clocks = (clock_counter_sequential_end - clock_counter_sequential_start) / (double)(clock_counter_SIMD_end - clock_counter_SIMD_start);
    double speedup_time = time_seq / time_simd;

    //Il parallelismo ideale (P) in questo contesto è dato dal numero di elementi che 
    //l'istruzione SIMD può processare contemporaneamente.
    //Le istruzioni SSE (Streaming SIMD Extensions) lavorano tipicamente con registri da 128 bit (__m128i). 
    //Poiché usiamo char (che sono tipicamente 8 bit o 1 byte) per rappresentare le celle della griglia
    int ideal_parallelism = 16; // 128 bit / 8 bit per char
    double efficiency_clocks = (speedup_clocks / ideal_parallelism);
    double efficiency_time = (speedup_time / ideal_parallelism);

    // Liberazione della memoria
    aligned_free_grid(grid_a);
    aligned_free_grid(grid_b);

    printf("Elapsed clocks (SIMD): %lu\n", clock_counter_SIMD_end-clock_counter_SIMD_start);
    printf("Tempo di esecuzione (SIMD): %f ms\n", time_simd*1000);
    printf("-----------------------\n");
    printf("Elapsed clocks (Sequenziale): %lu\n", clock_counter_sequential_end-clock_counter_sequential_start);
    printf("Tempo di esecuzione (Sequenziale): %f ms\n", time_seq*1000);
    printf("-----------------------\n");
    printf("Speed-up (clocks) = %3.2f\n", speedup_clocks*1.0);
    printf("Speed-up (time) = %3.2f\n", speedup_time*1.0);
    printf("Ideal Parallelism (P) = %d (128 bits / 8 bits)\n", ideal_parallelism);
    printf("Efficiency (clocks) = %f\n", efficiency_clocks);
    printf("Efficiency (time) = %f\n", efficiency_time);
    
    return 0;
}