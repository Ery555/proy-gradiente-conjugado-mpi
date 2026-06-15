#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 4  // Tamaño de la matriz (Para probar en red, usaremos múltiplos de 4)

int main(int argc, char** argv) {
    int procesadores, procesador;
    
    // Matriz y vectores globales (Solo el Procesador 0 los creará completos)
    double A[N][N];
    double b[N], x[N], r[N], p[N], w[N];
    
    // Cada procesador recibirá un bloque de filas. 
    // Si N=4 y hay 4 máquinas, cada uno recibe 1 fila.
    int filas_por_proc; 
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &procesadores);
    MPI_Comm_rank(MPI_COMM_WORLD, &procesador);

    filas_por_proc = N / procesadores;
    
    // Espacio de memoria local para las filas que le tocan a cada máquina
    double A_local[filas_por_proc][N];
    double w_local[filas_por_proc];

    // ==========================================
    // PASO 1: ENTRADA DE DATOS (PROCESADOR 0)
    // ==========================================
    if (procesador == 0) {
        // Generar Matriz Aleatoria Simétrica Definida Positiva
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i == j) {
                    A[i][j] = N * 10.0 + (rand() % 10); // Diagonal dominante
                } else {
                    A[i][j] = (rand() % 5) + 1.0;
                    A[j][i] = A[i][j]; // Asegura simetría
                }
            }
            b[i] = (rand() % 10) + 1.0; // Vector b aleatorio
            x[i] = 0.0;                 // Aproximación inicial x_0 = 0
        }
        
        printf("[Master] Matriz y vector b generados con éxito.\n");
    }

    // ==========================================
    // PASO 2: DISTRIBUIR LOS DATOS POR LA RED
    // ==========================================
    // Tu licenciado te enseñó MPI_Bcast y MPI_Scatter. Aquí los usamos:
    
    // 1. Enviamos el vector b y la aproximación x a TODOS los procesadores
    MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 2. Cortamos la matriz A por filas y mandamos un pedazo a cada máquina
    MPI_Scatter(A, filas_por_proc * N, MPI_DOUBLE, A_local, filas_por_proc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ==========================================
    // PASO 3: PREPARACIÓN DEL GRADIENTE CONJUGADO
    // ==========================================
    // Cada máquina calcula localmente el residuo inicial r = b - A*x
    // Como x inicial es cero, r_0 = b, y el vector de dirección p_0 = r_0
    for(int i = 0; i < N; i++) {
        r[i] = b[i];
        p[i] = r[i]; 
    }

    // Variables de control del método matemático
    double alpha, beta, numerador_r, denominador_p, numerador_r_nuevo;
    int k = 0;
    int max_iteraciones = N; // El método converge en máximo N pasos

    // ==========================================
    // PASO 4: EL BUCLE PARALELO DISTRIBUIDO
    // ==========================================
    while (k < max_iteraciones) {
        
        // --- PRODUCTO MATRIZ-VECTOR PARALELO (Requisito III.2) ---
        // Cada máquina multiplica SOLO las filas de la matriz que recibió (A_local)
        // por el vector de dirección actual (p). El resultado parcial se guarda en w_local.
        for (int i = 0; i < filas_por_proc; i++) {
            w_local[i] = 0.0;
            for (int j = 0; j < N; j++) {
                w_local[i] += A_local[i][j] * p[j];
            }
        }

        // Reunimos los pedazos calculados por cada máquina en el vector global 'w'
        // Usamos MPI_Allgather para que todas las máquinas tengan el vector 'w' actualizado al mismo tiempo
        MPI_Allgather(w_local, filas_por_proc, MPI_DOUBLE, w, filas_por_proc, MPI_DOUBLE, MPI_COMM_WORLD);

        // --- CÁLCULOS MATEMÁTICOS DE ALFA Y BETA ---
        // Numerador: r^T * r
        numerador_r = 0.0;
        for(int i = 0; i < N; i++) numerador_r += r[i] * r[i];

        // Denominador: p^T * w  (donde w = A * p)
        denominador_p = 0.0;
        for(int i = 0; i < N; i++) denominador_p += p[i] * w[i];

        // Factor de paso alfa
        alpha = numerador_r / denominador_p;

        // Actualizar el vector solución x y el residuo r
        for(int i = 0; i < N; i++) {
            x[i] = x[i] + alpha * p[i];
            r[i] = r[i] - alpha * w[i];
        }

        // Nuevo numerador con el residuo actualizado
        numerador_r_nuevo = 0.0;
        for(int i = 0; i < N; i++) numerador_r_nuevo += r[i] * r[i];

        // Criterio de parada: Si el residuo es casi cero, ya encontramos la solución exacta
        if (sqrt(numerador_r_nuevo) < 1e-6) {
            break;
        }

        // Factor beta
        beta = numerador_r_nuevo / numerador_r;

        // Actualizar vector de dirección p para la siguiente vuelta
        for(int i = 0; i < N; i++) {

            p[i] = r[i] + beta * p[i];
        }

        k++;
    }

    // ==========================================
    // PASO 5: RECOLECCIÓN Y MUESTRA DE RESULTADOS
    // ==========================================
    if (procesador == 0) {
        printf("\n====================================\n");
        printf("SISTEMA RESUELTO EN COMPUTACIÓN DISTRIBUIDA\n");
        printf("Iteraciones totales: %d\n", k + 1);
        printf("Vector Solución X resultante:\n");
        for (int i = 0; i < N; i++) {
            printf("x[%d] = %f\n", i, x[i]);
        }
        printf("====================================\n");
    }

    MPI_Finalize();
    return 0;
}
