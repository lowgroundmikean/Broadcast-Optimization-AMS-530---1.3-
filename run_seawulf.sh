#!/usr/bin/env bash
#
# run_seawulf.slurm -- AMS 530 Problem 1.3 benchmark
#
# Submit with:   sbatch run_seawulf.slurm
# Watch with:    squeue -u $USER
# Output lands in bcast32_<jobid>.out
#
# BEFORE SUBMITTING, check two things against SeaWulf's docs, because they
# differ between clusters and I could not verify them:
#   1. the partition name on the -p line below   (run: sinfo)
#   2. the MPI module name in the module load    (run: module avail)
# SeaWulf also has two sets of login nodes that reach different compute nodes,
# so submit from the one matching the partition you choose.
#
#SBATCH --job-name=bcast32
#SBATCH --output=bcast32_%j.out
#SBATCH --nodes=2
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=16
#SBATCH --time=00:20:00
#SBATCH -p short-40core

set -e

module purge
module load slurm
module load mvapich2/gcc/2.3.7     # CHECK: `module avail` for the right name

echo "host      : $(hostname)"
echo "nodes     : $SLURM_JOB_NUM_NODES"
echo "tasks     : $SLURM_NTASKS"
echo "started   : $(date)"
echo

mpicc -O2 -Wall -std=c99 -o bcast32 bcast32.c -lm

# ----------------------------------------------------------------------------
# Run 1: the payload the problem specifies.
# ----------------------------------------------------------------------------
echo "############ 16 doubles (the assigned payload) ############"
mpirun -np 32 ./bcast32 16 2000
echo

# ----------------------------------------------------------------------------
# Run 2: sweep the message size. Small messages are latency-bound, so the
# round count dominates and the 6-round schedule should beat the 31-round
# linear baseline by roughly 5x. Large messages are bandwidth-bound, so the
# advantage narrows. The crossover is worth a paragraph in the report.
# ----------------------------------------------------------------------------
echo "############ message-size sweep ############"
for n in 1 16 128 1024 8192 65536 262144; do
    echo "---- count = $n doubles ($((n * 8)) bytes) ----"
    mpirun -np 32 ./bcast32 "$n" 500
    echo
done

# ----------------------------------------------------------------------------
# Run 3: all 32 ranks on ONE node. Every link is then shared memory rather than
# the interconnect. Comparing against the 2-node run above separates the cost
# of the schedule itself from the cost of crossing the network, since the
# schedule is oblivious to which links happen to be remote.
# ----------------------------------------------------------------------------
echo "############ 32 ranks on a single node (intra-node) ############"
srun --nodes=1 --ntasks=32 --ntasks-per-node=32 ./bcast32 16 2000 || \
    echo "(single-node run skipped: needs a node with >= 32 cores)"

echo
echo "finished  : $(date)"
