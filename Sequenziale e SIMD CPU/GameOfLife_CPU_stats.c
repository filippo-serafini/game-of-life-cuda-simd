#define _POSIX_C_SOURCE 199309L // Richiesto per clock_gettime
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <immintrin.h> // Per gli intrinsics SSE

// Aggiungi la libreria time.h per il timing POSIX
#if defined(__APPLE__) || defined(__linux__)
#include <time.h>
#else
// Mantieni windows.h solo per Windows
#include <windows.h>
#endif

// --- Costanti ---
#define LOGIC_SIZE 2048       // Dimensione effettiva della griglia N x N
#define PADDED_SIZE (LOGIC_SIZE + 2) // N+2 x N+2 con zero-padding
#define ALIGNMENT 16        // Allineamento richiesto da SSE

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

// --- Memoria Allineata e Libera ---

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
        ptr = (char*)malloc(size); // Fallback non allineato
    #endif

    if (ptr != NULL) {
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

// --- Funzioni di Inizializzazione e Stampa (senza modifiche) ---

// Inizializza la griglia con un Glider (Versione SIMD/Padded)
void initialize_glider(char* grid_data) {
    int r = 10;
    int c = 10;

    // Griglia[riga * PADDED_SIZE + colonna]
    grid_data[r * PADDED_SIZE + c + 1] = 1;
    grid_data[(r + 1) * PADDED_SIZE + c + 2] = 1;
    grid_data[(r + 2) * PADDED_SIZE + c] = 1;
    grid_data[(r + 2) * PADDED_SIZE + c + 1] = 1;
    grid_data[(r + 2) * PADDED_SIZE + c + 2] = 1;
}

// Inizializza la griglia con valori 0 o 1 in modo deterministico (Versione SIMD/Padded)
void init_random_reproducible(char* grid, int width, int height, unsigned int seed) {
    // Azzeramento di TUTTA la griglia (con padding) 
    // gia' fatto nella malloc allineata
    
    // Probabilità che sia 0 o 1
    float probability = 0.5;
    // Seed riproducibile
    srand(seed);

    // Inizializza solo la zona logica (1..LOGIC_SIZE) saltando il padding
    for (int i = 1; i <= LOGIC_SIZE; ++i) {
        for (int j = 1; j <= LOGIC_SIZE; ++j) {
            // Genera un float tra 0.0 e 1.0
            float r = (float)(rand()) / (float)(RAND_MAX);
            // Se r è minore della probabilità (es. 0.5), la cella è viva (1), altrimenti morta (0)
            grid[i * PADDED_SIZE + j] = (r < probability) ? 1 : 0;
        }
    }
}

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

// --- Funzione Principale di Calcolo con SSE (senza modifiche) ---
void update_with_sse(char* current_grid, char* next_grid) {
    // Caricamento costante dei vettori di confronto fuori dal loop
    const __m128i three_vec = _mm_set1_epi8(3); // 0x03 per ogni byte
    const __m128i two_vec = _mm_set1_epi8(2);   // 0x02 per ogni byte
    const __m128i one_vec = _mm_set1_epi8(1);   // 0x01 per ogni byte

    // Loop sulle righe logiche
    for (int i = 1; i <= LOGIC_SIZE; ++i) {
        char* row_prev = current_grid + (i - 1) * PADDED_SIZE;
        char* row_curr = current_grid + (i) * PADDED_SIZE;
        char* row_next = current_grid + (i + 1) * PADDED_SIZE;

        // Loop sulle colonne logiche in blocchi di 16 celle (j = 1 a LOGIC_SIZE)
        for (int j = 1; j <= LOGIC_SIZE; j += 16) {

            // Caricamenti sovrapposti per la riga precedente
            __m128i prev_left  = _mm_loadu_si128((__m128i*)(row_prev + j - 1));
            __m128i prev_mid   = _mm_loadu_si128((__m128i*)(row_prev + j));
            __m128i prev_right = _mm_loadu_si128((__m128i*)(row_prev + j + 1));

            // Caricamenti sovrapposti per la riga corrente
            __m128i cur_left   = _mm_loadu_si128((__m128i*)(row_curr + j - 1));
            __m128i cur_mid    = _mm_loadu_si128((__m128i*)(row_curr + j));
            __m128i cur_right  = _mm_loadu_si128((__m128i*)(row_curr + j + 1));

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

            // Applicazione delle Regole: Nuovo Stato = (N=3) OR (Stato Attuale AND N=2)
            __m128i is_three = _mm_cmpeq_epi8(neigh_alive, three_vec);
            __m128i is_two = _mm_cmpeq_epi8(neigh_alive, two_vec);

            __m128i new_state_mask = _mm_or_si128(
                is_three,
                _mm_and_si128(cur_mid, is_two)
            );

            // Conversione in binario dello stato calcolato (0 o 1)
            __m128i next_cells = _mm_and_si128(new_state_mask, one_vec);

            // 5. Memorizzazione del nuovo stato
            char* target_ptr_next = next_grid + i * PADDED_SIZE + j;
            _mm_storeu_si128((__m128i*)target_ptr_next, next_cells);
        }
    }
}

// -------------------------------------------- Versione Sequenziale (senza modifiche) --------------------------------------------

/* Alloca una matrice (righe x cols) contigua */
static char* allocate_grid() {

    char *g = malloc((size_t)LOGIC_SIZE * LOGIC_SIZE * sizeof(char));
    if (!g) {
        fprintf(stderr, "Errore: allocazione memoria fallita\n");
        exit(EXIT_FAILURE);
    }
    return g;
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

// Inizializza la griglia con valori 0 o 1 in modo deterministico (Versione Sequenziale)
void init_random_reproducible_sequential(char* grid, int width, int height, unsigned int seed) {
    
    // Probabilità che sia 0 o 1
    float probability = 0.5;
    // Seed riproducibile
    srand(seed);

    int total_cells = width * height;

    for (int i = 0; i < total_cells; ++i) {
        // Genera un float tra 0.0 e 1.0
        float r = (float)(rand()) / (float)(RAND_MAX);
        
        // Se r è minore della probabilità (es. 0.5), la cella è viva (1), altrimenti morta (0)
        grid[i] = (r < probability) ? 1 : 0;
    }
}

// Funzione di aggiornamento sequenziale
static void update_sequential(char* curr_grid, char* next_grid){

    // La versione sequenziale deve usare gli indici della griglia NON-PADDED per funzionare correttamente
    // Poiché il tuo codice sequenziale usa gli indici PADDED (idx = row_offset + col) e
    // alloca la memoria PADDED solo con la malloc allineata (che poi liberi),
    // rendiamo la versione sequenziale consistente con la struttura PADDED per semplificare.

    for (int row = 1; row <= LOGIC_SIZE; ++row) {
        int row_offset = row * PADDED_SIZE;

        for (int col = 1; col <= LOGIC_SIZE; ++col) {

            int idx = row_offset + col; // Indice della cella corrente

            // Conta i vicini sommando direttamente gli offset fissi
            int n =
                curr_grid[idx - PADDED_SIZE - 1] +
                curr_grid[idx - PADDED_SIZE]     +
                curr_grid[idx - PADDED_SIZE + 1] +
                curr_grid[idx - 1]               +
                curr_grid[idx + 1]               +
                curr_grid[idx + PADDED_SIZE - 1] +
                curr_grid[idx + PADDED_SIZE]     +
                curr_grid[idx + PADDED_SIZE + 1];

            char is_alive = curr_grid[idx];

            next_grid[idx] = (n == 3) | (is_alive & (n == 2));
        }
    }
}

// --- Main Program ---
int main() {
    printf("Game of Life (%dx%d) C con SSE SIMD\n", LOGIC_SIZE, LOGIC_SIZE);

    //varibili per tempo sequenziale
    double time_seq_start, time_seq_end, time_seq;
    double time_seq_start_tot, time_seq_end_tot, time_seq_tot;
    //variabili per tempo simd
    double time_simd_start, time_simd_end, time_simd;
    double time_simd_start_tot, time_simd_end_tot, time_simd_tot;

    // Variabili Windows per i contatori ad alta risoluzione
#if defined(_WIN32)
    LARGE_INTEGER frequency_win;
    QueryPerformanceFrequency(&frequency_win);
#endif

    int generations = 15;

    // --------------------------------------- SEQUENZIALE ---------------------------------------

    // ISTANTE DI INIZIO TOTALE SEQUENZIALE
    #if defined(__APPLE__) || defined(__linux__)
        time_seq_start_tot = get_time_posix();
    #else // Windows
        LARGE_INTEGER start_win_seq_tot;
        QueryPerformanceCounter(&start_win_seq_tot);
        time_seq_start_tot = (double)start_win_seq_tot.QuadPart;
    #endif

    // Alloca e inizializza le due griglie allineate (usiamo le funzioni allineate anche per la seq
    // per coerenza e per usare update_sequential con gli indici PADDED)
    char* grid_a = aligned_malloc_grid();
    char* grid_b = aligned_malloc_grid();

    init_random_reproducible_sequential(grid_a, LOGIC_SIZE, LOGIC_SIZE, 42);

    char* current = grid_a;
    char* next = grid_b;

    // ISTANTE DI INIZIO CALCOLO GENERAZIONI SEQUENZIALE
    #if defined(__APPLE__) || defined(__linux__)
        time_seq_start = get_time_posix();
    #else // Windows
        LARGE_INTEGER start_win_seq;
        QueryPerformanceCounter(&start_win_seq);
        time_seq_start = (double)start_win_seq.QuadPart;
    #endif

    for (int g = 0; g < generations; ++g) {
        update_sequential(current, next);

        // Scambia le griglie (doppio buffering)
        char* temp = current;
        current = next;
        next = temp;
    }

    // ISTANTE FINALE CALCOLO GENERAZIONI SEQUENZIALE
    #if defined(__APPLE__) || defined(__linux__)
        time_seq_end = get_time_posix();
        time_seq = time_seq_end - time_seq_start;
    #else // Windows
        LARGE_INTEGER end_win_seq;
        QueryPerformanceCounter(&end_win_seq);
        time_seq_end = (double)end_win_seq.QuadPart;
        time_seq = (time_seq_end - time_seq_start) / frequency_win.QuadPart;
    #endif

    aligned_free_grid(grid_a);
    aligned_free_grid(grid_b);

    // ISTANTE FINALE TOTALE SEQUENZIALE
    #if defined(__APPLE__) || defined(__linux__)
        time_seq_end_tot = get_time_posix();
        time_seq_tot = time_seq_end_tot - time_seq_start_tot;
    #else // Windows
        LARGE_INTEGER end_win_seq_tot;
        QueryPerformanceCounter(&end_win_seq_tot);
        time_seq_end_tot = (double)end_win_seq_tot.QuadPart;
        time_seq_tot = (time_seq_end_tot - time_seq_start_tot) / frequency_win.QuadPart;
    #endif

    // ------------------------ SIMD ------------------------

    // ISTANTE DI INIZIO TOTALE SIMD
    #if defined(__APPLE__) || defined(__linux__)
        time_simd_start_tot = get_time_posix();
    #else // Windows
        LARGE_INTEGER start_win_simd_tot;
        QueryPerformanceCounter(&start_win_simd_tot);
        time_simd_start_tot = (double)start_win_simd_tot.QuadPart;
    #endif

    // Alloca e inizializza le due griglie
    grid_a = aligned_malloc_grid();
    grid_b = aligned_malloc_grid();

    if (!grid_a || !grid_b) {
        fprintf(stderr, "Errore nell'allocazione della memoria allineata.\n");
        return 1;
    }

    init_random_reproducible(grid_a, LOGIC_SIZE, LOGIC_SIZE, 42);

    current = grid_a;
    next = grid_b;

    // ISTANTE DI INIZIO CALCOLO GENERAZIONI SIMD
    #if defined(__APPLE__) || defined(__linux__)
        time_simd_start = get_time_posix();
    #else // Windows
        LARGE_INTEGER start_win_simd;
        QueryPerformanceCounter(&start_win_simd);
        time_simd_start = (double)start_win_simd.QuadPart;
    #endif

    for (int g = 0; g < generations; ++g) {
        update_with_sse(current, next);

        // Scambia le griglie (doppio buffering)
        char* temp = current;
        current = next;
        next = temp;
    }

    // ISTANTE FINALE CALCOLO GENERAZIONI SIMD
    #if defined(__APPLE__) || defined(__linux__)
        time_simd_end = get_time_posix();
        time_simd = time_simd_end - time_simd_start;
    #else // Windows
        LARGE_INTEGER end_win_simd;
        QueryPerformanceCounter(&end_win_simd);
        time_simd_end = (double)end_win_simd.QuadPart;
        time_simd = (time_simd_end - time_simd_start) / frequency_win.QuadPart;
    #endif

    // Liberazione della memoria
    aligned_free_grid(grid_a);
    aligned_free_grid(grid_b);

    // ISTANTE FINALE TOTALE SIMD
    #if defined(__APPLE__) || defined(__linux__)
        time_simd_end_tot = get_time_posix();
        time_simd_tot = time_simd_end_tot - time_simd_start_tot;
    #else // Windows
        LARGE_INTEGER end_win_simd_tot;
        QueryPerformanceCounter(&end_win_simd_tot);
        time_simd_end_tot = (double)end_win_simd_tot.QuadPart;
        time_simd_tot = (time_simd_end_tot - time_simd_start_tot) / frequency_win.QuadPart;
    #endif

    // ------------------------ CALCOLO STATISTICHE ------------------------
    double speedup_time = time_seq / time_simd;
    double speedup_time_tot = time_seq_tot / time_simd_tot;

    // Il parallelismo ideale (P) è 16 per SSE (128 bit) e char (8 bit)
    int ideal_parallelism = 16;
    double efficiency_time = (speedup_time / ideal_parallelism);
    double efficiency_time_tot = (speedup_time_tot / ideal_parallelism);

    // ------------------------ STAMPA RISULTATI ------------------------
    printf("Tempo di esecuzione totale (Sequenziale): %f ms\n", time_seq_tot * 1000);
    printf("Tempo di esecuzione generazioni (Sequenziale): %f ms\n", time_seq * 1000);
    printf("-----------------------\n");
    printf("Tempo di esecuzione totale (SIMD): %f ms\n", time_simd_tot * 1000);
    printf("Tempo di esecuzione generazioni (SIMD): %f ms\n", time_simd * 1000);
    printf("-----------------------\n");
    printf("Speed-up totale = %3.2f\n", speedup_time_tot * 1.0);
    printf("Speed-up generazioni = %3.2f\n", speedup_time * 1.0);
    printf("Ideal Parallelism (P) = %d (128 bits / 8 bits)\n", ideal_parallelism);
    printf("Efficiency totale = %f (o %3.2f%%)\n", efficiency_time_tot, efficiency_time_tot * 100.0);
    printf("Efficiency generazioni = %f (o %3.2f%%)\n", efficiency_time, efficiency_time * 100.0);

    return 0;
}
