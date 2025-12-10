#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <immintrin.h>

#ifdef _WIN32
#include <intrin.h>
#include <windows.h>
#else
#include <x86intrin.h>
#endif

// --- Costanti ---
#define LOGIC_SIZE 2048
#define GENERAZIONI 15
#define PADDED_SIZE (LOGIC_SIZE + 2)
#define ALIGNMENT 16

// --- Funzioni per Lettura dei Clock della CPU ---

// Lettura diretta del Time Stamp Counter (RDTSC)
// Restituisce il numero di clock della CPU dal boot
static inline uint64_t rdtsc(void) {
    return __rdtsc();
}

// Ottiene la frequenza della CPU in Hz
// Su Windows, utilizza QueryPerformanceFrequency per ottenere una stima accurata
static uint64_t get_cpu_frequency(void) {
    #if defined(_WIN32)
        // Su Windows, usiamo QueryPerformanceFrequency che è altamente accurato
        LARGE_INTEGER freq;
        QueryPerformanceFrequency(&freq);
        return freq.QuadPart;
    #else
        // Estrazione della frequenza da /proc/cpuinfo su Linux
        FILE *fp = fopen("/proc/cpuinfo", "r");
        if (!fp) {
            return 2400000000ULL; // Default fallback: 2.4 GHz
        }
        
        double freq = 0;
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            if (sscanf(line, "cpu MHz : %lf", &freq) == 1) {
                fclose(fp);
                return (uint64_t)(freq * 1000000);
            }
        }
        fclose(fp);
        return 2400000000ULL; // Default fallback
    #endif
}

// --- Memoria Allineata ---
// Utilizziamo questa funzione per allocare memoria allineata a 16 byte
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
        ptr = (char*)malloc(size);
    #endif

    if (ptr != NULL) {
        memset(ptr, 0, size);
    }
    return ptr;
}
// Funzione per liberare la memoria allocata con aligned_malloc_grid
void aligned_free_grid(char* ptr) {
    #if defined(_WIN32)
        _aligned_free(ptr);
    #else
        free(ptr);
    #endif
}

// --- Inizializzazione ---
// Inizializza la griglia con valori 0 o 1 in modo deterministico
void init_random_reproducible(char* grid, unsigned int seed) {
    
    float probability = 0.5;
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
// --- Funzione di Aggiornamento SIMD ---
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

            // Regola 1: Nuova vita o Sopravvivenza (N=3): 
            // Crea una maschera dove i byte sono 0xFF (vero) se il conteggio dei vicini è 3.
            __m128i is_three = _mm_cmpeq_epi8(neigh_alive, three_vec);
            // Regola 2: Sopravvivenza (N=2): 
            // Crea una maschera dove i byte sono 0xFF se il conteggio è 2.
            __m128i is_two = _mm_cmpeq_epi8(neigh_alive, two_vec);
            // Combinazione delle Regole: La maschera dello stato futuro è: 
            // (N=3) OR (Stato Attuale AND N=2). La cella attuale è contenuta nel vettore centrale della riga corrente (cur_mid).
            __m128i new_state_mask = _mm_or_si128(
                is_three,
                _mm_and_si128(cur_mid, is_two)
            );

            // Conversione in Binario: Le maschere di confronto contengono 0xFF per "vero" e 0x00 per "falso".
            // Per ottenere lo stato finale come 1 o 0, si esegue un AND logico 
            // con il vettore costante one_vec (contenente 0x01 per ogni byte):
            __m128i next_cells = _mm_and_si128(new_state_mask, one_vec);

            // Memorizzazione del nuovo stato della griglia di gioco
            char* target_ptr_next = next_grid + i * PADDED_SIZE + j;
            _mm_storeu_si128((__m128i*)target_ptr_next, next_cells);
        }
    }
}

// Funzione di aggiornamento sequenziale
static void update_sequential(char* curr_grid, char* next_grid){
    // La versione sequenziale di seguito è consistente con la struttura della griglia con zero-padding,
    // come definito nella nostra relazione
    for (int row = 1; row <= LOGIC_SIZE; ++row) {
        int row_offset = row * PADDED_SIZE; // Offset della riga corrente dovuto al padding

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

int main() {
    printf("Game of Life (%dx%d) %d-generazioni C con SSE SIMD\n", LOGIC_SIZE, LOGIC_SIZE, GENERAZIONI);
    printf("Misurazione basata su conteggio dei clock CPU\n\n");

    // Ottieniamo la frequenza della CPU
    uint64_t cpu_freq = get_cpu_frequency();
    printf("Frequenza CPU rilevata: %.2f GHz\n", (double)cpu_freq / 1e9);
    printf("================================================\n\n");

    // Variabili per clock sequenziale
    uint64_t clk_seq_start, clk_seq_end, clk_seq;
    uint64_t clk_seq_start_tot, clk_seq_end_tot, clk_seq_tot;
    // Variabili per clock SIMD
    uint64_t clk_simd_start, clk_simd_end, clk_simd;
    uint64_t clk_simd_start_tot, clk_simd_end_tot, clk_simd_tot;

    // ======================== SEQUENZIALE ========================

    printf("Esecuzione versione SEQUENZIALE...\n");

    // INIZIO TOTALE SEQUENZIALE
    clk_seq_start_tot = rdtsc();

    // Alloca le due griglie allineate (usiamo le funzioni allineate anche per la versione
    // sequenziale per coerenza e per usare update_sequential con gli indici PADDED)
    char* grid_a = aligned_malloc_grid();
    char* grid_b = aligned_malloc_grid();
    if (!grid_a || !grid_b) {
        fprintf(stderr, "Errore nell'allocazione della memoria.\n");
        return 1;
    }
    // Inizializza la griglia di partenza
    init_random_reproducible(grid_a, 42);

    char* current = grid_a;
    char* next = grid_b;

    // INIZIO CALCOLO GENERAZIONI SEQUENZIALE
    clk_seq_start = rdtsc();

    for (int g = 0; g < GENERAZIONI; ++g) {
        update_sequential(current, next);
        char* temp = current;
        current = next;
        next = temp;
    }

    // FINE CALCOLO GENERAZIONI SEQUENZIALE
    clk_seq_end = rdtsc();
    clk_seq = clk_seq_end - clk_seq_start;

    aligned_free_grid(grid_a);
    aligned_free_grid(grid_b);

    // FINE TOTALE SEQUENZIALE
    clk_seq_end_tot = rdtsc();
    clk_seq_tot = clk_seq_end_tot - clk_seq_start_tot;

    // ======================== SIMD ========================

    printf("Esecuzione versione SIMD...\n");

    // INIZIO TOTALE SIMD
    clk_simd_start_tot = rdtsc();

    // Alloca e inizializza le due griglie
    grid_a = aligned_malloc_grid();
    grid_b = aligned_malloc_grid();

    if (!grid_a || !grid_b) {
        fprintf(stderr, "Errore nell'allocazione della memoria.\n");
        return 1;
    }

    init_random_reproducible(grid_a, 42);

    current = grid_a;
    next = grid_b;

    // INIZIO CALCOLO GENERAZIONI SIMD
    clk_simd_start = rdtsc();

    for (int g = 0; g < GENERAZIONI; ++g) {
        update_with_sse(current, next);
        char* temp = current;
        current = next;
        next = temp;
    }

    // FINE CALCOLO GENERAZIONI SIMD
    clk_simd_end = rdtsc();
    clk_simd = clk_simd_end - clk_simd_start;

    aligned_free_grid(grid_a);
    aligned_free_grid(grid_b);

    // FINE TOTALE SIMD
    clk_simd_end_tot = rdtsc();
    clk_simd_tot = clk_simd_end_tot - clk_simd_start_tot;

    // ======================== STATISTICHE ========================

    // Converte i clock in tempo reale (ms)
    double time_seq_ms = (double)clk_seq / (double)cpu_freq * 1000.0;
    double time_seq_tot_ms = (double)clk_seq_tot / (double)cpu_freq * 1000.0;
    double time_simd_ms = (double)clk_simd / (double)cpu_freq * 1000.0;
    double time_simd_tot_ms = (double)clk_simd_tot / (double)cpu_freq * 1000.0;

    // Calcola speedup
    double speedup_time = time_seq_ms / time_simd_ms;
    double speedup_time_tot = time_seq_tot_ms / time_simd_tot_ms;

    // Parallelismo ideale: 16 per SSE (128 bit / 8 bit)
    int ideal_parallelism = 16;
    double efficiency_time = (speedup_time / ideal_parallelism) * 100.0;
    double efficiency_time_tot = (speedup_time_tot / ideal_parallelism) * 100.0;

    // ======================== STAMPA RISULTATI ========================

    printf("\n================================================\n");
    printf("RISULTATI MISURAZIONE (basati su clock CPU)\n");
    printf("================================================\n\n");

    printf("[SEQUENZIALE]\n");
    printf("  Clock totali: %llu\n", clk_seq_tot);
    printf("  Clock generazioni: %llu\n", clk_seq);
    printf("  Tempo totale: %.6f ms\n", time_seq_tot_ms);
    printf("  Tempo generazioni: %.6f ms\n", time_seq_ms);

    printf("\n[SIMD]\n");
    printf("  Clock totali: %llu\n", clk_simd_tot);
    printf("  Clock generazioni: %llu\n", clk_simd);
    printf("  Tempo totale: %.6f ms\n", time_simd_tot_ms);
    printf("  Tempo generazioni: %.6f ms\n", time_simd_ms);

    printf("\n[SPEEDUP]\n");
    printf("  Speedup totale: %.2fx\n", speedup_time_tot);
    printf("  Speedup generazioni: %.2fx\n", speedup_time);

    printf("\n[EFFICIENZA]\n");
    printf("  Efficienza totale: %.2f%%\n", efficiency_time_tot);
    printf("  Efficienza generazioni: %.2f%%\n", efficiency_time);

    printf("\n================================================\n");

    return 0;
}