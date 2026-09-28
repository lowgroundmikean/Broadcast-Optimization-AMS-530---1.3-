/*
 * placement.c -- Measure the actual rank-to-node mapping and recompute the
 * intra/inter-node breakdown of the broadcast schedule from it.
 *
 * Section 5.3 of the report asserts that 9 of the 31 scheduled links cross the
 * node boundary. That figure was originally derived by assuming Slurm's default
 * block distribution. This program measures the placement instead of assuming
 * it, so the claim rests on data.
 *
 * Build: mpicc -O2 -Wall -std=c99 -o placement placement.c
 * Run:   mpirun -np 32 ./placement
 */

#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "arrays.h"

int main(int argc, char **argv)
{
    int rank, size, i, t;
    char myname[MPI_MAX_PROCESSOR_NAME];
    char *all = NULL;
    int namelen;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != NV) {
        if (rank == 0) fprintf(stderr, "needs exactly %d ranks\n", NV);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    memset(myname, 0, sizeof myname);
    MPI_Get_processor_name(myname, &namelen);

    if (rank == 0) all = (char *) malloc((size_t) size * MPI_MAX_PROCESSOR_NAME);
    MPI_Gather(myname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
               all, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        char hosts[64][MPI_MAX_PROCESSOR_NAME];
        int nhosts = 0, hostof[NV];

        /* build the list of distinct hosts and map each vertex onto one */
        for (i = 0; i < size; i++) {
            char *n = all + (size_t) i * MPI_MAX_PROCESSOR_NAME;
            int h, found = -1;
            for (h = 0; h < nhosts; h++)
                if (strcmp(hosts[h], n) == 0) { found = h; break; }
            if (found < 0) { strcpy(hosts[nhosts], n); found = nhosts++; }
            hostof[i] = found;          /* index by rank; vertex v is rank v-1 */
        }

        printf("MEASURED PLACEMENT\n");
        printf("  distinct nodes: %d\n", nhosts);
        for (i = 0; i < nhosts; i++) {
            int c = 0, r;
            printf("  node %c = %-20s vertices:", 'A' + i, hosts[i]);
            for (r = 0; r < size; r++)
                if (hostof[r] == i) { printf(" %d", r + 1); c++; }
            printf("   (%d ranks)\n", c);
        }

        printf("\nSCHEDULE vs MEASURED PLACEMENT\n");
        printf("%-7s%-7s%-12s%-12s%-10s\n", "round", "links", "intra-node",
               "cross-node", "% cross");
        int tot_links = 0, tot_cross = 0;
        for (t = 1; t <= NSTEPS; t++) {
            int n = 0, x = 0;
            for (i = 0; i < NLINKS; i++) {
                if (SCHEDULE[i].step != t) continue;
                n++;
                if (hostof[SCHEDULE[i].from - 1] != hostof[SCHEDULE[i].to - 1]) x++;
            }
            tot_links += n; tot_cross += x;
            printf("%-7d%-7d%-12d%-12d%6.0f%%\n", t, n, n - x, x, 100.0 * x / n);
        }
        printf("%-7s%-7d%-12d%-12d%6.0f%%\n", "total", tot_links,
               tot_links - tot_cross, tot_cross, 100.0 * tot_cross / tot_links);

        /* direction of each crossing, for the Finding 3 table */
        if (nhosts == 2) {
            printf("\nDIRECTION OF INTER-NODE TRAFFIC PER ROUND\n");
            printf("%-7s%-8s%-8s%s\n", "round", "A->B", "B->A", "crossing links");
            for (t = 1; t <= NSTEPS; t++) {
                int ab = 0, ba = 0;
                printf("%-7d", t);
                char buf[256]; buf[0] = '\0';
                for (i = 0; i < NLINKS; i++) {
                    if (SCHEDULE[i].step != t) continue;
                    int hf = hostof[SCHEDULE[i].from - 1];
                    int ht = hostof[SCHEDULE[i].to - 1];
                    if (hf == ht) continue;
                    if (hf == 0) ab++; else ba++;
                    char one[32];
                    sprintf(one, "%s%d-%d", buf[0] ? ", " : "",
                            SCHEDULE[i].from, SCHEDULE[i].to);
                    strcat(buf, one);
                }
                printf("%-8d%-8d%s\n", ab, ba, buf[0] ? buf : "none");
            }
        }
        free(all);
    }

    MPI_Finalize();
    return 0;
}
