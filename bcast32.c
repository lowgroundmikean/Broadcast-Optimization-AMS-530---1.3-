/*
 * bcast32.c -- Optimal broadcast of 16 doubles on the (32,3) regular graph.
 *
 * AMS 530 Project 1, Problem 1.3.
 *
 * Each MPI rank is one vertex of the graph: vertex v maps to rank v-1, so the
 * designated source (vertex 1) is rank 0.
 *
 * The broadcast follows a hand-derived schedule of S = 6 rounds. Six rounds is
 * optimal in the 1-port (telephone) model, in which each vertex takes part in
 * at most one communication per round. The bound comes from a Fibonacci-type
 * recursion: if h(t) is the largest number of vertices reachable in t rounds
 * from a vertex holding two free ports (one edge having been spent receiving),
 * then h(t) = 1 + h(t-1) + h(t-2). The source has three free ports, so at most
 * g(t) = 1 + h(t-1) + h(t-2) + h(t-3) vertices can be informed in t rounds.
 * g(5) = 24 < 32 and g(6) = 40 >= 32, so no cubic graph on 32 vertices can be
 * broadcast in fewer than six rounds. The schedule below attains six, and in
 * fact matches g(t) exactly at every intermediate round.
 *
 * The program
 *   1. validates the schedule against the graph at startup,
 *   2. runs the broadcast and verifies every rank received the payload,
 *   3. times it against MPI_Bcast and a naive linear broadcast,
 *   4. optionally reports a per-round timing breakdown.
 *
 * No deadlock is possible: within each round the set of senders and the set of
 * receivers are disjoint, so every MPI_Send has a matching MPI_Recv already
 * posted or about to be posted by a different rank. The program therefore does
 * not depend on eager-protocol buffering for correctness, and behaves the same
 * for payloads above the eager threshold.
 *
 * Build:  mpicc -O2 -Wall -o bcast32 bcast32.c
 * Run:    mpirun -np 32 ./bcast32 [count] [reps]
 *         mpirun --oversubscribe -np 32 ./bcast32      (fewer physical cores)
 *
 *   count  doubles to broadcast   (default 16, as the problem specifies)
 *   reps   timed repetitions      (default 1000)
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "arrays.h"

#define TAG 1

/* ---------- helpers ---------------------------------------------------- */

/* Is {u,v} an edge of the graph? Vertices are 1-based. */
static int is_edge(int u, int v)
{
    int i;
    for (i = 0; i < NEDGES; i++) {
        if ((EDGES[i][0] == u && EDGES[i][1] == v) ||
            (EDGES[i][0] == v && EDGES[i][1] == u)) {
            return 1;
        }
    }
    return 0;
}

/*
 * Check the schedule obeys the telephone model on this graph. Run on rank 0
 * only. Returns the number of violations found, printing each one.
 */
static int validate_schedule(void)
{
    int informed[NV + 1], recvd[NV + 1];
    int sends_in_step[NV + 1], recvs_in_step[NV + 1];
    int v, i, step, bad = 0;

    for (v = 1; v <= NV; v++) { informed[v] = 0; recvd[v] = 0; }
    informed[SOURCE] = 1;

    for (step = 1; step <= NSTEPS; step++) {
        for (v = 1; v <= NV; v++) { sends_in_step[v] = 0; recvs_in_step[v] = 0; }

        for (i = 0; i < NLINKS; i++) {
            if (SCHEDULE[i].step != step) continue;
            int a = SCHEDULE[i].from, b = SCHEDULE[i].to;

            if (!is_edge(a, b)) {
                printf("  step %d: %d-%d is not an edge\n", step, a, b); bad++;
            }
            if (!informed[a]) {
                printf("  step %d: sender %d is not yet informed\n", step, a); bad++;
            }
            if (informed[b]) {
                printf("  step %d: receiver %d is already informed\n", step, b); bad++;
            }
            sends_in_step[a]++;
            recvs_in_step[b]++;
            recvd[b]++;
        }

        /* 1-port model: at most one communication per vertex per round, and no
           vertex may both send and receive in the same round. */
        for (v = 1; v <= NV; v++) {
            if (sends_in_step[v] > 1) {
                printf("  step %d: vertex %d sends %d times\n", step, v, sends_in_step[v]); bad++;
            }
            if (recvs_in_step[v] > 1) {
                printf("  step %d: vertex %d receives %d times\n", step, v, recvs_in_step[v]); bad++;
            }
            if (sends_in_step[v] && recvs_in_step[v]) {
                printf("  step %d: vertex %d both sends and receives\n", step, v); bad++;
            }
        }
        for (v = 1; v <= NV; v++) {
            if (recvs_in_step[v]) informed[v] = 1;
        }
    }

    for (v = 1; v <= NV; v++) {
        if (v == SOURCE) {
            if (recvd[v] != 0) { printf("  source received %d times\n", recvd[v]); bad++; }
        } else if (recvd[v] != 1) {
            printf("  vertex %d received %d times (expected 1)\n", v, recvd[v]); bad++;
        }
        if (!informed[v]) { printf("  vertex %d never informed\n", v); bad++; }
    }
    return bad;
}

/* ---------- the three broadcast implementations ------------------------ */

/* The optimal 6-round schedule. */
static void graph_bcast(double *buf, int count, int rank)
{
    int step, i;
    for (step = 1; step <= NSTEPS; step++) {
        for (i = 0; i < NLINKS; i++) {
            if (SCHEDULE[i].step != step) continue;
            int from = SCHEDULE[i].from - 1;   /* vertex -> rank */
            int to   = SCHEDULE[i].to   - 1;
            if (rank == from) {
                MPI_Send(buf, count, MPI_DOUBLE, to, TAG, MPI_COMM_WORLD);
            } else if (rank == to) {
                MPI_Recv(buf, count, MPI_DOUBLE, from, TAG, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            }
        }
    }
}

/* Same schedule, timed round by round. Barriers isolate the rounds, so the
   totals here exceed graph_bcast: the barrier cost is measurement overhead. */
static void graph_bcast_timed(double *buf, int count, int rank, double *step_time)
{
    int step, i;
    for (step = 1; step <= NSTEPS; step++) {
        double t0;
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (i = 0; i < NLINKS; i++) {
            if (SCHEDULE[i].step != step) continue;
            int from = SCHEDULE[i].from - 1;
            int to   = SCHEDULE[i].to   - 1;
            if (rank == from) {
                MPI_Send(buf, count, MPI_DOUBLE, to, TAG, MPI_COMM_WORLD);
            } else if (rank == to) {
                MPI_Recv(buf, count, MPI_DOUBLE, from, TAG, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            }
        }
        step_time[step - 1] += MPI_Wtime() - t0;
    }
}

/* Naive baseline: the source sends to all 31 others in turn, 31 rounds. */
static void linear_bcast(double *buf, int count, int rank)
{
    int r;
    if (rank == SOURCE - 1) {
        for (r = 0; r < NV; r++) {
            if (r == SOURCE - 1) continue;
            MPI_Send(buf, count, MPI_DOUBLE, r, TAG, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(buf, count, MPI_DOUBLE, SOURCE - 1, TAG, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
}

/* ---------- payload handling ------------------------------------------- */

static void fill_source(double *buf, int count)
{
    int i;
    for (i = 0; i < count; i++) buf[i] = 1000.0 + 0.5 * i;
}

static void clear_buf(double *buf, int count, int rank)
{
    int i;
    if (rank == SOURCE - 1) { fill_source(buf, count); return; }
    for (i = 0; i < count; i++) buf[i] = -1.0;   /* poison */
}

static int check_buf(const double *buf, int count)
{
    int i;
    for (i = 0; i < count; i++) {
        if (fabs(buf[i] - (1000.0 + 0.5 * i)) > 1e-12) return 1;
    }
    return 0;
}

/* Barrier, then time one call; return the slowest rank's elapsed time. */
static double timed_call(void (*fn)(double *, int, int),
                         double *buf, int count, int rank)
{
    double t0, local, global;
    clear_buf(buf, count, rank);
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    fn(buf, count, rank);
    local = MPI_Wtime() - t0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return global;
}

static void mpi_bcast_wrapper(double *buf, int count, int rank)
{
    (void) rank;
    MPI_Bcast(buf, count, MPI_DOUBLE, SOURCE - 1, MPI_COMM_WORLD);
}

/* ---------- main -------------------------------------------------------- */

int main(int argc, char **argv)
{
    int rank, size, count = 16, reps = 1000;
    int rep, s, errs, total_errs;
    double *buf;
    double t, sum_graph = 0, sum_mpi = 0, sum_lin = 0;
    double min_graph = 1e30, min_mpi = 1e30, min_lin = 1e30;
    double step_time[NSTEPS], step_max[NSTEPS];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != NV) {
        if (rank == 0) {
            fprintf(stderr, "This program needs exactly %d ranks, one per vertex "
                            "(got %d).\n", NV, size);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (argc > 1) count = atoi(argv[1]);
    if (argc > 2) reps  = atoi(argv[2]);
    if (count < 1 || reps < 1) {
        if (rank == 0) fprintf(stderr, "count and reps must be positive\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* 1. schedule validation */
    if (rank == 0) {
        int bad;
        printf("(32,3) graph broadcast, S = %d rounds, %d ranks\n", NSTEPS, size);
        printf("payload %d doubles (%zu bytes), %d timed repetitions\n\n",
               count, count * sizeof(double), reps);
        printf("schedule validation against the graph:\n");
        bad = validate_schedule();
        printf("  %s\n\n", bad ? "FAILED" : "all telephone-model rules satisfied, "
                                            "31 edges, every vertex informed once");
        if (bad) { MPI_Abort(MPI_COMM_WORLD, 2); }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    buf = (double *) malloc((size_t) count * sizeof(double));
    if (!buf) { MPI_Abort(MPI_COMM_WORLD, 3); }

    /* 2. correctness */
    clear_buf(buf, count, rank);
    graph_bcast(buf, count, rank);
    errs = check_buf(buf, count);
    MPI_Reduce(&errs, &total_errs, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("correctness: %d of %d ranks hold the wrong payload\n\n",
               total_errs, size);
    }

    /* warm-up, so the first timed rep does not pay connection setup */
    for (rep = 0; rep < 20; rep++) {
        clear_buf(buf, count, rank); graph_bcast(buf, count, rank);
        clear_buf(buf, count, rank); mpi_bcast_wrapper(buf, count, rank);
        clear_buf(buf, count, rank); linear_bcast(buf, count, rank);
    }

    /* 3. timing */
    for (rep = 0; rep < reps; rep++) {
        t = timed_call(graph_bcast, buf, count, rank);
        sum_graph += t; if (t < min_graph) min_graph = t;

        t = timed_call(mpi_bcast_wrapper, buf, count, rank);
        sum_mpi += t;   if (t < min_mpi)   min_mpi = t;

        t = timed_call(linear_bcast, buf, count, rank);
        sum_lin += t;   if (t < min_lin)   min_lin = t;
    }

    if (rank == 0) {
        printf("%-34s %12s %12s %10s\n", "broadcast", "mean (us)", "min (us)", "rounds");
        printf("%-34s %12s %12s %10s\n", "---------", "---------", "--------", "------");
        printf("%-34s %12.3f %12.3f %10d\n", "graph schedule (this work)",
               1e6 * sum_graph / reps, 1e6 * min_graph, NSTEPS);
        printf("%-34s %12.3f %12.3f %10s\n", "MPI_Bcast (library)",
               1e6 * sum_mpi / reps, 1e6 * min_mpi, "-");
        printf("%-34s %12.3f %12.3f %10d\n", "linear (naive baseline)",
               1e6 * sum_lin / reps, 1e6 * min_lin, NV - 1);
        printf("\nspeedup of graph schedule over linear : %.2fx\n",
               sum_lin / sum_graph);
        printf("ratio graph schedule to MPI_Bcast     : %.2fx\n",
               sum_graph / sum_mpi);
    }

    /* 4. per-round breakdown (includes barrier overhead; see comment above) */
    for (s = 0; s < NSTEPS; s++) step_time[s] = 0.0;
    for (rep = 0; rep < reps; rep++) {
        clear_buf(buf, count, rank);
        graph_bcast_timed(buf, count, rank, step_time);
    }
    MPI_Reduce(step_time, step_max, NSTEPS, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const int links_per_step[NSTEPS] = {1, 2, 4, 6, 10, 8};
        int informed = 1;
        printf("\nper-round breakdown (barrier-separated, so totals exceed the above)\n");
        printf("%-6s %10s %12s %14s\n", "round", "links", "informed", "mean (us)");
        for (s = 0; s < NSTEPS; s++) {
            informed += links_per_step[s];
            printf("%-6d %10d %12d %14.3f\n", s + 1, links_per_step[s], informed,
                   1e6 * step_max[s] / reps);
        }
    }

    free(buf);
    MPI_Finalize();
    return 0;
}
