#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

unsigned int seed_global;

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
        z  = sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
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
        par[i].m  = rnd01() * 0.01 * (ncside*ncside) / n_part / G * EPSILON2;
        par[i].active = 1;
    }
}

static inline double periodic_diff(double d, double side) {
    if      (d >  side/2) return d - side;
    else if (d < -side/2) return d + side;
    else                  return d;
}

void simulate(particle_t *local, particle_t *all, long long n_part,
              long local_n, double side, int n_steps, int *total_collisions,
              int rank, int size)
{
    *total_collisions = 0;
    for (int step = 0; step < n_steps; step++) {
        // reúne todas as partículas em todos os processos
        MPI_Allgather(local, local_n*sizeof(particle_t), MPI_BYTE,
                      all,   local_n*sizeof(particle_t), MPI_BYTE,
                      MPI_COMM_WORLD);

        // cálculo da força sobre cada partícula local
        for (long i = 0; i < local_n; i++) {
            if (!local[i].active) continue;
            double fx = 0, fy = 0;
            for (long j = 0; j < n_part; j++) {
                if (!all[j].active || (rank*local_n + i) == j) continue;
                double dx = periodic_diff(all[j].x - local[i].x, side);
                double dy = periodic_diff(all[j].y - local[i].y, side);
                double dist2 = dx*dx + dy*dy;
                if (dist2 > 1e-12) {
                    double dist = sqrt(dist2);
                    double f    = G * local[i].m * all[j].m / dist2;
                    fx += f * dx / dist;
                    fy += f * dy / dist;
                }
            }
            local[i].vx += (fx / local[i].m) * DELTAT;
            local[i].vy += (fy / local[i].m) * DELTAT;
        }

        // atualização de posições
        for (long i = 0; i < local_n; i++) {
            if (!local[i].active) continue;
            local[i].x += local[i].vx * DELTAT;
            local[i].y += local[i].vy * DELTAT;
            if (local[i].x < 0)      local[i].x += side;
            else if (local[i].x >= side) local[i].x -= side;
            if (local[i].y < 0)      local[i].y += side;
            else if (local[i].y >= side) local[i].y -= side;
        }

        // detecção de colisões
        int local_coll = 0;
        for (long i = 0; i < local_n; i++) {
            if (!local[i].active) continue;
            for (long j = 0; j < n_part; j++) {
                if (!all[j].active) continue;
                if ((rank*local_n + i) == j) continue;
                double dx = periodic_diff(local[i].x - all[j].x, side);
                double dy = periodic_diff(local[i].y - all[j].y, side);
                if (dx*dx + dy*dy < EPSILON2) {
                    local[i].active = 0;
                    all[j].active   = 0;
                    local_coll += 2;
                    break;
                }
            }
        }

        // soma global de colisões deste passo
        int step_coll = 0;
        MPI_Reduce(&local_coll, &step_coll, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) *total_collisions += step_coll;
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 6) {
        if (rank == 0)
            fprintf(stderr, "Uso: %s <semente> <lado> <ncside> <n_part> <n_steps>\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    long seed      = atol(argv[1]);
    double side    = atof(argv[2]);
    int ncside     = atoi(argv[3]);
    long long n_part = atoll(argv[4]);
    int n_steps    = atoi(argv[5]);

    long local_n = n_part / size;
    particle_t *local    = malloc(local_n * sizeof(particle_t));
    particle_t *all      = malloc(n_part  * sizeof(particle_t));

    if (rank == 0) {
        particle_t *init = malloc(n_part * sizeof(particle_t));
        init_particles(seed, side, ncside, n_part, init);
        MPI_Scatter(init, local_n*sizeof(particle_t), MPI_BYTE,
                    local, local_n*sizeof(particle_t), MPI_BYTE, 0, MPI_COMM_WORLD);
        free(init);
    } else {
        MPI_Scatter(NULL, local_n*sizeof(particle_t), MPI_BYTE,
                    local, local_n*sizeof(particle_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    }

    double t0 = -omp_get_wtime();
    int total_collisions = 0;

    simulate(local, all, n_part, local_n, side, n_steps, &total_collisions, rank, size);

    double elapsed = t0 + omp_get_wtime();
    if (rank == 0) {
        fprintf(stderr, "%.1fs\n", elapsed);
        // imprime exatamente as duas linhas de saída
        printf("%.3f %.3f\n", all[0].x, all[0].y);
        printf("%d\n", total_collisions);
    }

    free(local);
    free(all);
    MPI_Finalize();
    return EXIT_SUCCESS;
}

