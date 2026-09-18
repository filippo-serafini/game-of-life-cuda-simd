#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <immintrin.h> // Include per gli intrinsics SSE

// --- Costanti ---
#define LOGIC_SIZE 64*5       // Dimensione effettiva della griglia N x N
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
    
    // Loop sulle righe logiche (i = 1 a 64)
    for (int i = 1; i <= LOGIC_SIZE; ++i) { 
        // puntatori all'inizio della riga precedente, corrente e successiva
        char* row_prev = current_grid + (i - 1) * PADDED_SIZE; 
        char* row_curr = current_grid + (i) * PADDED_SIZE;
        char* row_next = current_grid + (i + 1) * PADDED_SIZE;
        
        // Loop sulle colonne logiche (j = 1 a 64) in blocchi di 16 celle.
        // Poiché 64 è multiplo di 16, non servono controlli sul bordo destro.
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

/* Conta vicini viventi attorno alla cella (r,c).
   Bordi non avvolgenti: celle esterne contate come morte */
static int count_neighbors(const char *g, int r, int c) {
    int count = 0;
    for (int dr = -1; dr <= 1; ++dr) {
        int rr = r + dr;
        if (rr < 0 || rr >= LOGIC_SIZE) continue;
        for (int dc = -1; dc <= 1; ++dc) {
            int cc = c + dc;
            if (cc < 0 || cc >= LOGIC_SIZE) continue;
            if (dr == 0 && dc == 0) continue; // stessa cella
            count += g[rr * LOGIC_SIZE + cc] ? 1 : 0;
        }
    }
    return count;
}

static void update_sequential(char* curr_grid, char* next_grid){
    for (int r = 0; r < LOGIC_SIZE; ++r) {
        for (int c = 0; c < LOGIC_SIZE; ++c) {
            int n = count_neighbors(curr_grid, r, c);
            char alive = curr_grid[r * LOGIC_SIZE + c];
            char next = 0;
            if (alive) {
                // Sopravvive con 2 o 3 vicini, altrimenti muore
                next = (n == 2 || n == 3) ? 1 : 0;
            } else {
                // Nasce se esattamente 3 vicini
                next = (n == 3) ? 1 : 0;
            }
            next_grid[r * LOGIC_SIZE + c] = next;
        }
    }
}
        
// --- Main Program ---
int main() {
    printf("Game of Life (64x64) C con SSE SIMD\n");

    // Istanti di inizio e fine per l'esecuzione
    uint64_t clock_counter_sequential_start, clock_counter_sequential_end;
    uint64_t clock_counter_SIMD_start, clock_counter_SIMD_end;
    
    int generations = 15; // 5mila

    // --------------------------------------- SEQUENZIALE ---------------------------------------
    
    // Alloca e inizializza le due griglie
    char* grid_a = allocate_grid();
    char* grid_b = allocate_grid();

    initialize_glider_sequential(grid_a);
    initialize_b(grid_b);

    char* current = grid_a;
    char* next = grid_b;

    // ISTANTE DI INIZIO
    clock_counter_sequential_start = __rdtsc(); 

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

    free(grid_a);
    free(grid_b);

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

    // Liberazione della memoria
    aligned_free_grid(grid_a);
    aligned_free_grid(grid_b);

    printf("Elapsed clocks (SIMD): %lu\n", clock_counter_SIMD_end-clock_counter_SIMD_start);
    printf("Elapsed clocks (Sequential): %lu\n", clock_counter_sequential_end-clock_counter_sequential_start);
    printf("Speed-up = %3.2f\n", (clock_counter_sequential_end - clock_counter_sequential_start)/((clock_counter_SIMD_end - clock_counter_SIMD_start)*1.0));

    return 0;
}