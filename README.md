# AMS 530 Project 1 — Problem 1.3

Optimal broadcast of 16 double-precision values on the optimal (N = 32, k = 3)
regular graph of Zhang, Xu and Deng, implemented in MPI.

**Result:** the broadcast completes in **S = 6 rounds**, activating
**1, 2, 4, 6, 10, 8** edges per round for a total of 31. Six rounds is optimal:
a Fibonacci-type counting bound shows that no cubic graph on 32 vertices can be
broadcast in fewer, and the schedule attains the bound at every intermediate
round, not only at termination.

## Contents

| File | Purpose |
|---|---|
| `AMS530_Project1_Problem1.3.pdf` | The report |
| `bcast32.c` | MPI implementation, validator and benchmark |
| `arrays.h` | The 48 edges and 31 scheduled links, generated from the adjacency matrix |
| `run_seawulf.sh` | Slurm batch script used to produce the measurements |
| `bcast32_90162.out` | Raw output of the benchmark run |
| `fig/` | Figures used in the report |

## Build and run

```bash
module load slurm
module load openmpi/gcc12.1/4.1.4
mpicc -O2 -Wall -std=c99 -o bcast32 bcast32.c -lm

# usage: mpirun -np 32 ./bcast32 [count] [reps]
#   count   doubles to broadcast   (default 16)
#   reps    timed repetitions      (default 1000)
```

On a cluster:

```bash
sbatch run_seawulf.sh        # 2 nodes, 32 tasks, partition short-28core
squeue -u $USER
cat bcast32_<jobid>.out
```

On a workstation, oversubscribed. Adequate for correctness, **not** for timing,
since all ranks then share the same cores:

```bash
mpirun --oversubscribe -np 32 ./bcast32
```

The program requires exactly 32 ranks, one per vertex, and aborts otherwise.

## What the program checks

Before timing anything it validates the schedule against the embedded graph:
every scheduled link is an edge; every sender is already informed; no receiver
is already informed; no vertex sends or receives twice within a round; no vertex
both sends and receives within a round; and each of the 31 non-source vertices
receives exactly once. It then verifies correctness by poisoning every
non-source buffer and confirming all 32 ranks hold the expected payload.

## Measurements

SeaWulf, partition `short-28core`, 2 nodes × 16 tasks, Open MPI 4.1.4 / GCC 12.1.

| Payload | Graph schedule | MPI_Bcast | Linear | vs linear |
|---|---|---|---|---|
| 128 B (assigned) | 9.42 µs | 6.04 µs | 11.19 µs | 1.19x |
| 64 KB | 159.15 µs | 106.68 µs | 522.04 µs | 3.28x |
| 2 MB | 3919.53 µs | 2844.11 µs | 11384.82 µs | 2.90x |

Three findings, developed in the report:

1. At the assigned payload the expected five-fold advantage over the linear
   baseline collapses to 1.07x, because 128-byte messages fall below the eager
   threshold and the linear broadcast's 31 sends overlap instead of serialising.
   The advantage recovers to 3.28x once messages enter the rendezvous regime.
2. `MPI_Bcast` is uniformly 1.35x–1.60x faster. It is topology-aware and crosses
   the inter-node link once; a schedule derived from graph structure alone
   crosses it nine times out of 31 links.
3. Per-round cost is governed by the *direction* of inter-node traffic, not the
   number of active links. Round 5 carries 10 links but only 534 µs, because its
   two crossings run in opposite directions on a full-duplex link; rounds with
   two same-direction crossings cost roughly 930–955 µs.

## References

1. Y. Zhang, Z. Xu, Y. Deng, "A Structured Table of Graphs with Symmetries and
   Other Special Properties," *Symmetry* 12(1), art. 2, 2020. arXiv:1910.13539.
2. S. M. Hedetniemi, S. T. Hedetniemi, A. L. Liestman, "A Survey of Gossiping and
   Broadcasting in Communication Networks," *Networks* 18(4), 319–349, 1988.
3. P. J. Slater, E. J. Cockayne, S. T. Hedetniemi, "Information Dissemination in
   Trees," *SIAM J. Computing* 10(4), 692–701, 1981.
