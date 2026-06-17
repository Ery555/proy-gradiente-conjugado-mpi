#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char** argv) {
    int procesadores, procesador;
    
    // 1. EL TAMAÑO N AHORA ES DINÁMICO (Por defecto 1024, o se pasa por consola)
    int N = 1024; 
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &procesadores);
    MPI_Comm_rank(MPI_COMM_WORLD, &procesador);

    // Control de seguridad
    if (N % procesadores != 0) {
        if (procesador == 0) {
            printf("[ERROR] El tamaño N (%d) debe ser divisible exactamente entre el numero de procesos (%d).\n", N, procesadores);
        }
        MPI_Finalize();
        return 1;
    }

    int filas_por_proc = N / procesadores;
    
    // 2. ASIGNACIÓN DINÁMICA DE MEMORIA (MALLOC)
    // Usamos arreglos 1D para garantizar que la memoria sea contigua para MPI
    double *A = NULL;
    if (procesador == 0) {
        A = (double*)malloc(N * N * sizeof(double));
    }
    
    double *b = (double*)malloc(N * sizeof(double));
    double *x = (double*)malloc(N * sizeof(double));
    double *r = (double*)malloc(N * sizeof(double));
    double *p = (double*)malloc(N * sizeof(double));
    double *w = (double*)malloc(N * sizeof(double));
    
    // Memoria local de cada máquina
    double *A_local = (double*)malloc(filas_por_proc * N * sizeof(double));
    double *w_local = (double*)malloc(filas_por_proc * sizeof(double));

    // ==========================================
    // PASO 1: ENTRADA DE DATOS (PROCESADOR 0)
    // ==========================================
    if (procesador == 0) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j <= i; j++) {
                if (i == j) {
                    A[i * N + j] = N * 10.0 + (rand() % 10); // Diagonal dominante
                } else {
                    double valor = (rand() % 5) + 1.0;
                    A[i * N + j] = valor;
                    A[j * N + i] = valor; // Simetría
                }
            }
            b[i] = (rand() % 10) + 1.0; 
            x[i] = 0.0;                 
        }
        printf("[Master] Matriz simetrica de %dx%d generada en RAM.\n", N, N);
    }

    // ==========================================
    // PASO 2: DISTRIBUIR LOS DATOS POR LA RED
    // ==========================================
    MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(A, filas_por_proc * N, MPI_DOUBLE, A_local, filas_por_proc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ==========================================
    // PASO 3: PREPARACIÓN DEL GRADIENTE CONJUGADO
    // ==========================================
    for(int i = 0; i < N; i++) {
        r[i] = b[i];
        p[i] = r[i]; 
    }

    double alpha, beta, numerador_r, denominador_p, numerador_r_nuevo;
    int k = 0;
    int max_iteraciones = N; 

    // --- ARRANQUE DEL CRONÓMETRO ---
    // MPI_Barrier asegura que nadie inicie el reloj hasta que todos tengan sus datos
    MPI_Barrier(MPI_COMM_WORLD);
    double tiempo_inicio = MPI_Wtime();

    // ==========================================
    // PASO 4: EL BUCLE PARALELO DISTRIBUIDO
    // ==========================================
    while (k < max_iteraciones) {
        
        // Producto matriz-vector paralelo
        for (int i = 0; i < filas_por_proc; i++) {
            w_local[i] = 0.0;
            for (int j = 0; j < N; j++) {
                // A_local es un bloque continuo, se accede linealmente
                w_local[i] += A_local[i * N + j] * p[j];
            }
        }

        MPI_Allgather(w_local, filas_por_proc, MPI_DOUBLE, w, filas_por_proc, MPI_DOUBLE, MPI_COMM_WORLD);

        // Operaciones algebraicas (redundantes por seguridad de tiempo)
        numerador_r = 0.0;
        for(int i = 0; i < N; i++) numerador_r += r[i] * r[i];

        denominador_p = 0.0;
        for(int i = 0; i < N; i++) denominador_p += p[i] * w[i];

        alpha = numerador_r / denominador_p;

        for(int i = 0; i < N; i++) {
            x[i] = x[i] + alpha * p[i];
            r[i] = r[i] - alpha * w[i];
        }

        numerador_r_nuevo = 0.0;
        for(int i = 0; i < N; i++) numerador_r_nuevo += r[i] * r[i];

        if (sqrt(numerador_r_nuevo) < 1e-6) {
            break;
        }

        beta = numerador_r_nuevo / numerador_r;

        for(int i = 0; i < N; i++) {
            p[i] = r[i] + beta * p[i];
        }

        k++;
    }

    // --- FIN DEL CRONÓMETRO ---
    MPI_Barrier(MPI_COMM_WORLD);
    double tiempo_fin = MPI_Wtime();

    // ==========================================
    // PASO 5: RECOLECCIÓN Y MUESTRA DE RESULTADOS
    // ==========================================
    if (procesador == 0) {
        double tiempo_total = tiempo_fin - tiempo_inicio;
        printf("\n====================================\n");
        printf("SISTEMA RESUELTO EN CLUSTER DISTRIBUIDO\n");
        printf("Nodos utilizados : %d\n", procesadores);
        printf("Tamano matriz (N): %d\n", N);
        printf("Iteraciones      : %d\n", (k < max_iteraciones) ? k + 1 : k);
        printf("TIEMPO DE COMPUTO: %f segundos\n", tiempo_total);
        printf("====================================\n");

        // Solo imprimimos el vector si la matriz es pequeña para no saturar la terminal
        if (N <= 16) {
            printf("Vector Solucion X resultante:\n");
            for (int i = 0; i < N; i++) {
                printf("x[%d] = %f\n", i, x[i]);
            }
            printf("====================================\n");
        } else {
            printf("(Vector solucion omitido en consola por ser muy grande)\n");
            printf("====================================\n");
        }
    }

    // 3. LIBERACIÓN DE MEMORIA DINÁMICA
    if (procesador == 0) {
        free(A);
    }
    free(b); free(x); free(r); free(p); free(w);
    free(A_local); free(w_local);

    MPI_Finalize();
    return 0;
}
