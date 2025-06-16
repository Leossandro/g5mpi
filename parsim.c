/*
 * parsim.c – Implementação serial do simulador de partículas
 * CPD 2024-25, Trabalho Acadêmico - Simulação de Partículas
 *
 * 1. Introdução
 * O propósito deste trabalho acadêmico é ganhar experiência em programação paralela
 * em sistemas UMA (Uniform Memory Access) e multicomputadores, usando OpenMP e
 * MPI, respectivamente. Para tal, os estudantes devem escrever uma implementação
 * serial e duas paralelas de um simulador de partículas em movimento no espaço livre.
 *
 * 2. Descrição do Problema
 * Consideramos um cenário onde as partículas se movem livremente no espaço e as
 * únicas forças às quais elas são submetidas são devidas à gravidade umas das outras.
 * Para simplificar, assuma um espaço 2D, um quadrado com lado 1000 (as unidades de
 * medida reais não são relevantes).
 */

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define G 6.67408e-11
#define EPSILON 0.005
#define EPSILON2 (EPSILON*EPSILON)
#define DELTAT 0.1

typedef struct {
    double x, y;
    double vx, vy;
    double m;
    int active;
} particle_t;

typedef struct {
    double mass;
    double com_x;
    double com_y;
} cell_t;

unsigned int seed_global;

/* Inicializa RNG */
void init_r4uni(int input_seed) {
    seed_global = input_seed + 987654321;
}

double rnd_uniform01() {
    int seed_in = seed_global;
    seed_global ^= (seed_global << 13);
    seed_global ^= (seed_global >> 17);
    seed_global ^= (seed_global << 5);
    return 0.5 + 0.2328306e-09 * (seed_in + (int)seed_global);
}

double rnd_normal01() {
    double u1, u2, z, result;
    do {
        u1 = rnd_uniform01();
        u2 = rnd_uniform01();
        z = sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
        result = 0.5 + 0.15 * z;
    } while (result < 0 || result >= 1);
    return result;
}

void init_particles(long seed, double side, long ncside, long long n_part, particle_t *par) {
    double (*rnd01)() = rnd_uniform01;
    if (seed < 0) {
        rnd01 = rnd_normal01;
        seed = -seed;
    }
    init_r4uni((int)seed);
    for (long long i = 0; i < n_part; i++) {
        par[i].x = rnd01() * side;
        par[i].y = rnd01() * side;
        par[i].vx = (rnd01() - 0.5) * side / ncside / 5.0;
        par[i].vy = (rnd01() - 0.5) * side / ncside / 5.0;
        par[i].m = rnd01() * 0.01 * (ncside * ncside) / n_part / G * EPSILON2;
        par[i].active = 1;
    }
}

static inline int mod(int a, int m) {
    int r = a % m;
    return r < 0 ? r + m : r;
}

static inline double periodic_diff(double d, double side) {
    if (d > side/2) return d - side;
    if (d < -side/2) return d + side;
    return d;
}

void simulation(particle_t *par, long long n_part, double side, int ncside, int n_steps, int *collision_count) {
    int n_cells = ncside * ncside;
    double cell_size = side / ncside;
    cell_t *cells = malloc(n_cells * sizeof(cell_t));
    int *cell_idx = malloc(n_part * sizeof(int));
    *collision_count = 0;

    for (int step = 0; step < n_steps; step++) {
        // 1. Zerar células
        for (int i = 0; i < n_cells; i++) {
            cells[i].mass = cells[i].com_x = cells[i].com_y = 0.0;
        }
        // 2. Atribuir partículas às células e acumular dados
        for (long long i = 0; i < n_part; i++) {
            if (!par[i].active) continue;
            int cx = mod((int)(par[i].x / cell_size), ncside);
            int cy = mod((int)(par[i].y / cell_size), ncside);
            int idx = cy * ncside + cx;
            cell_idx[i] = idx;
            cells[idx].mass += par[i].m;
            cells[idx].com_x += par[i].m * par[i].x;
            cells[idx].com_y += par[i].m * par[i].y;
        }
        for (int i = 0; i < n_cells; i++) {
            if (cells[i].mass > 0) {
                cells[i].com_x /= cells[i].mass;
                cells[i].com_y /= cells[i].mass;
            }
        }
        // 3. Forças e velocidades
        for (long long i = 0; i < n_part; i++) {
            if (!par[i].active) continue;
            double fx = 0.0, fy = 0.0;
            int cx = cell_idx[i] % ncside;
            int cy = cell_idx[i] / ncside;
            for (int iy = -1; iy <= 1; iy++) {
                for (int ix = -1; ix <= 1; ix++) {
                    int ncx = mod(cx + ix, ncside);
                    int ncy = mod(cy + iy, ncside);
                    int cidx = ncy * ncside + ncx;
                    if (cidx == cell_idx[i]) {
                        for (long long j = 0; j < n_part; j++) {
                            if (!par[j].active || j == i || cell_idx[j] != cidx) continue;
                            double dx = periodic_diff(par[j].x - par[i].x, side);
                            double dy = periodic_diff(par[j].y - par[i].y, side);
                            double dist2 = dx*dx + dy*dy;
                            if (dist2 > 1e-12) {
                                double dist = sqrt(dist2);
                                double f = G * par[i].m * par[j].m / dist2;
                                fx += f * dx / dist;
                                fy += f * dy / dist;
                            }
                        }
                    } else if (cells[cidx].mass > 0) {
                        double dx = periodic_diff(cells[cidx].com_x - par[i].x, side);
                        double dy = periodic_diff(cells[cidx].com_y - par[i].y, side);
                        double dist2 = dx*dx + dy*dy;
                        if (dist2 > 1e-12) {
                            double dist = sqrt(dist2);
                            double f = G * par[i].m * cells[cidx].mass / dist2;
                            fx += f * dx / dist;
                            fy += f * dy / dist;
                        }
                    }
                }
            }
            par[i].vx += (fx / par[i].m) * DELTAT;
            par[i].vy += (fy / par[i].m) * DELTAT;
        }
        // 4. Posições e índices
        for (long long i = 0; i < n_part; i++) {
            if (!par[i].active) continue;
            par[i].x += par[i].vx * DELTAT;
            par[i].y += par[i].vy * DELTAT;
            if (par[i].x < 0) par[i].x += side;
            if (par[i].x >= side) par[i].x -= side;
            if (par[i].y < 0) par[i].y += side;
            if (par[i].y >= side) par[i].y -= side;
            int cx = mod((int)(par[i].x / cell_size), ncside);
            int cy = mod((int)(par[i].y / cell_size), ncside);
            cell_idx[i] = cy * ncside + cx;
        }
        // 5. Colisões
        for (long long i = 0; i < n_part; i++) {
            if (!par[i].active) continue;
            for (long long j = i+1; j < n_part; j++) {
                if (!par[j].active || cell_idx[j] != cell_idx[i]) continue;
                double dx = periodic_diff(par[i].x - par[j].x, side);
                double dy = periodic_diff(par[i].y - par[j].y, side);
                if (dx*dx + dy*dy < EPSILON2) {
                    par[i].active = par[j].active = 0;
                    *collision_count += 2;
                }
            }
        }
    }
    free(cells);
    free(cell_idx);
}

int main(int argc, char *argv[]) {
    if (argc != 6) {
        fprintf(stderr, "Uso: %s <semente> <lado_simulacao> <ncside> <n_particulas> <n_passos>\n", argv[0]);
        return EXIT_FAILURE;
    }
    long seed = atol(argv[1]);
    double side = atof(argv[2]);
    int ncside = atoi(argv[3]);
    long long n_part = atoll(argv[4]);
    int n_steps = atoi(argv[5]);

    particle_t *particles = malloc(n_part * sizeof(particle_t));
    if (!particles) return EXIT_FAILURE;

    init_particles(seed, side, ncside, n_part, particles);

    double exec_time = -omp_get_wtime();
    int collision_count = 0;

    simulation(particles, n_part, side, ncside, n_steps, &collision_count);
    exec_time += omp_get_wtime();
    fprintf(stderr, "%.1fs\n", exec_time);

    printf("%.3f %.3f\n", particles[0].x, particles[0].y);
    printf("%d\n", collision_count);

    free(particles);
    return EXIT_SUCCESS;
}
