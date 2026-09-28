"""
make_figures.py -- Regenerate the four figures used in the report.

Self-contained: the graph, the schedule and the measured timings are embedded
below, so this reproduces the figures without any other file.

    pip install matplotlib networkx numpy
    python make_figures.py

Writes fig/fig1_tree.png .. fig/fig4_rounds.png at 200 dpi.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

os.makedirs("fig", exist_ok=True)

# ---------------------------------------------------------------- the graph
# Outer 32-cycle plus 16 chords, recovered from the adjacency matrix.
CHORDS = [(1, 12), (2, 23), (3, 19), (4, 26), (5, 30), (6, 13), (7, 18),
          (8, 24), (9, 31), (10, 20), (11, 27), (14, 21), (15, 25), (16, 32),
          (17, 28), (22, 29)]
EDGES = [(i, i % 32 + 1) for i in range(1, 33)] + CHORDS

# ------------------------------------------------------------- the schedule
SCHEDULE = [
    [(1, 12)],
    [(1, 2), (12, 13)],
    [(1, 32), (2, 23), (12, 11), (13, 14)],
    [(2, 3), (11, 27), (13, 6), (14, 21), (23, 24), (32, 16)],
    [(3, 4), (6, 7), (11, 10), (14, 15), (16, 17), (21, 20), (23, 22),
     (24, 8), (27, 26), (32, 31)],
    [(6, 5), (7, 18), (8, 9), (15, 25), (20, 19), (22, 29), (27, 28), (31, 30)],
]

# ------------------------------- measured on SeaWulf, 2 nodes x 16 tasks ---
PAYLOAD = np.array([8, 128, 1024, 8192, 65536, 524288, 2097152])
GRAPH = np.array([6.069, 9.419, 15.062, 39.578, 159.150, 998.903, 3919.527])
BCAST = np.array([3.795, 6.038, 10.511, 29.273, 106.675, 690.552, 2844.114])
LINEAR = np.array([8.598, 11.187, 32.799, 106.061, 522.035, 3022.658, 11384.818])
ROUND_2MB = np.array([346.3, 357.5, 940.9, 955.8, 533.6, 924.2])
LINKS_PER_ROUND = [1, 2, 4, 6, 10, 8]

COL = ["#2b2d6e", "#1f78b4", "#33a02c", "#e6ab02", "#e6550d", "#b2182b"]


def fig1_tree():
    G = nx.Graph(EDGES)
    pos = {v: (np.cos(np.pi / 2 - 2 * np.pi * (v - 1) / 32),
               np.sin(np.pi / 2 - 2 * np.pi * (v - 1) / 32)) for v in G}
    round_of = {frozenset(e): t for t, step in enumerate(SCHEDULE, 1) for e in step}

    fig, ax = plt.subplots(figsize=(7.2, 7.6))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="0.88", width=1.0)
    for t in range(1, 7):
        es = [tuple(e) for e, s in round_of.items() if s == t]
        nx.draw_networkx_edges(G, pos, edgelist=es, ax=ax, width=2.6,
                               edge_color=[COL[t - 1]] * len(es))
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_size=[280 if v == 1 else 165 for v in G],
                           node_color=["#d62728" if v == 1 else "white" for v in G],
                           edgecolors="k", linewidths=1.1)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7.5)
    handles = [plt.Line2D([0], [0], color=COL[t - 1], lw=2.6, label=f"round {t}")
               for t in range(1, 7)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01),
              ncol=6, fontsize=8.5, frameon=False)
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_title("The 31-edge broadcast tree on the (32,3) graph, coloured by round.\n"
                 "Grey edges are unused; vertex 1 (red) is the source.", fontsize=10)
    plt.tight_layout()
    plt.savefig("fig/fig1_tree.png", dpi=200, bbox_inches="tight")
    plt.close()


def fig2_time():
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(PAYLOAD, GRAPH, "o-", label="graph schedule (6 rounds)")
    ax.loglog(PAYLOAD, BCAST, "s-", label="MPI_Bcast (library)")
    ax.loglog(PAYLOAD, LINEAR, "^-", label="linear (31 rounds)")
    ax.set_xlabel("payload (bytes)")
    ax.set_ylabel("mean completion time (\u00b5s)")
    ax.set_title("Broadcast time vs payload, 32 ranks on 2 nodes", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("fig/fig2_time.png", dpi=200)
    plt.close()


def fig3_speedup():
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogx(PAYLOAD, LINEAR / GRAPH, "o-", label="vs linear baseline")
    ax.semilogx(PAYLOAD, BCAST / GRAPH, "s-", label="vs MPI_Bcast")
    ax.axhline(1, color="k", lw=0.8)
    ax.annotate("eager regime:\nlinear sends overlap", xy=(128, 1.19),
                xytext=(300, 1.9), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("rendezvous regime:\nlinear truly serial", xy=(65536, 3.28),
                xytext=(3000, 3.4), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("payload (bytes)")
    ax.set_ylabel("speedup of graph schedule")
    ax.set_title("Relative performance of the 6-round schedule", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("fig/fig3_speedup.png", dpi=200)
    plt.close()


def fig4_rounds():
    labels = ["0 cross", "0 cross", "2 same dir", "2 up 1 down",
              "1 each way", "2 same dir"]
    colours = ["#4c9f70", "#4c9f70", "#c0392b", "#c0392b", "#e39a2b", "#c0392b"]
    rounds = np.arange(1, 7)

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.bar(rounds, ROUND_2MB, color=colours)
    for r, t, lab in zip(rounds, ROUND_2MB, labels):
        ax.text(r, t + 18, lab, ha="center", fontsize=7.5)
    ax.set_xlabel("round")
    ax.set_ylabel("mean round time (\u00b5s)")
    ax.set_title("Per-round time at 2 MB is set by inter-node link contention,\n"
                 "not by the number of links", fontsize=10)
    ax.set_ylim(0, 1150)
    ax.grid(axis="y", alpha=0.3)
    sec = ax.twinx()
    sec.plot(rounds, LINKS_PER_ROUND, "ko--", ms=5, lw=1)
    sec.set_ylabel("links active in round")
    sec.set_ylim(0, 12)
    plt.tight_layout()
    plt.savefig("fig/fig4_rounds.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    fig1_tree()
    fig2_time()
    fig3_speedup()
    fig4_rounds()
    print("wrote:", sorted(os.listdir("fig")))
