# visualize_rgcn.py
# Creates a visualization of the relation graph used in MHD-v3 (RGCN input)
# Creates ALL overall cluster

# import argparse
# import networkx as nx
# import matplotlib.pyplot as plt
# import torch
# import json
# from pathlib import Path

# from src.utils.io import load_config
# from src.features.priors import load_atc_map, load_cyp_table
# from src.features.graph_relations import build_relation_graph

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, required=True, help="Path to YAML config (same as train_mhd_v3)")
#     parser.add_argument("--out", type=str, default="rgcn_graph.png", help="Output image filename")
#     parser.add_argument("--max_nodes", type=int, default=50, help="Subset of nodes to plot for clarity")
#     args = parser.parse_args()

#     cfg = load_config(args.config)
#     data_dir = Path(cfg["data"]["data_dir"])

#     # --- load priors
#     atc_map = load_atc_map(data_dir)
#     cyp_df = load_cyp_table(cfg["data"]["drug_cyp_file"])

#     # --- build relation graph
#     # pos_all edges not included here, but you can load datasets like in train_mhd_v3
#     dummy_drugs = list(atc_map.keys())[:args.max_nodes]
#     graph_data = build_relation_graph(
#         drugs=dummy_drugs,
#         atc_map=atc_map,
#         cyp_df=cyp_df,
#         ddi_edges=[]
#     )

#     edge_index = graph_data["edge_index"].cpu().numpy()
#     edge_type = graph_data["edge_type"].cpu().numpy()

#     # --- convert to networkx
#     G = nx.Graph()
#     for i in range(edge_index.shape[1]):
#         u, v = edge_index[:, i]
#         rel = int(edge_type[i])
#         G.add_edge(u, v, relation=rel)

#     # --- plot
#     pos = nx.spring_layout(G, seed=42)
#     rel_colors = {r: plt.cm.tab20(r % 20) for r in set(edge_type)}

#     plt.figure(figsize=(10, 10))
#     for (u, v, d) in G.edges(data=True):
#         nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=[rel_colors[d["relation"]]], width=1.5)
#     nx.draw_networkx_nodes(G, pos, node_size=300, node_color="lightblue")
#     nx.draw_networkx_labels(G, pos, font_size=6)
#     plt.title("RGCN Relation Graph (subset)")
#     plt.axis("off")
#     plt.savefig(args.out, dpi=300)
#     print(f"[INFO] Graph visualization saved -> {args.out}")

# if __name__ == "__main__":
#     main()


# -----------------------------------------------------------------------------------------------------------------------------#
# src/scripts/visualize_rgcn.py
# Visualize RGCN relation graph structure for sanity checking.
# Shows top 20 K values
# python -m src.scripts.visualize_rgcn --config src/config/exp_mhd_v3_cold.yaml --out outputs/rgcn_graph.png

# src/scripts/visualize_rgcn.py
# import argparse
# import numpy as np
# import pandas as pd
# import torch
# import matplotlib.pyplot as plt
# import networkx as nx

# from pathlib import Path
# from src.utils.io import load_config
# from src.data.ingest import load_chch, load_ddinter, load_decagon, merge_sources
# from src.features.priors import load_atc_map, load_cyp_table
# from src.features.graph_relations import build_relation_graph


# def plot_top_hubs(graph_data, drug2id, id2drug, out_path, top_k=20):
#     """Plot top-K hub drugs and their neighbors."""
#     edge_index = graph_data["edge_index"].cpu().numpy()
#     edge_type = graph_data["edge_type"].cpu().numpy()

#     # build graph with relation type attributes
#     G = nx.Graph()
#     for d_id, name in id2drug.items():
#         G.add_node(d_id, label=name)

#     for i in range(edge_index.shape[1]):
#         u, v = edge_index[:, i]
#         G.add_edge(u, v, rel=int(edge_type[i]))

#     # compute node degree
#     degrees = dict(G.degree())
#     hubs = sorted(degrees, key=degrees.get, reverse=True)[:top_k]

#     # induced subgraph: hubs + their neighbors
#     sub_nodes = set(hubs)
#     for h in hubs:
#         sub_nodes.update(G.neighbors(h))
#     H = G.subgraph(sub_nodes)

#     # layout
#     pos = nx.spring_layout(H, seed=42)
#     edge_colors = [d["rel"] for _, _, d in H.edges(data=True)]
#     cmap = plt.cm.tab20

#     # plot
#     plt.figure(figsize=(12, 12))
#     ax = plt.gca()
#     nx.draw_networkx_nodes(H, pos, node_size=30, node_color="blue", alpha=0.7, ax=ax)
#     nx.draw_networkx_edges(H, pos, edge_color=edge_colors, edge_cmap=cmap, width=0.8, alpha=0.5, ax=ax)

#     # add labels only for hubs (to avoid clutter)
#     labels = {n: id2drug[n] for n in hubs if n in H.nodes()}
#     nx.draw_networkx_labels(H, pos, labels=labels, font_size=8, font_color="black", ax=ax)

#     # colorbar for relation types
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(edge_colors), vmax=max(edge_colors)))
#     sm.set_array([])
#     plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04, label="Relation Types")

#     plt.title(f"Top-{top_k} Drug Hubs and Neighbors", fontsize=14)
#     plt.axis("off")
#     plt.savefig(out_path, dpi=300, bbox_inches="tight")
#     plt.close()

#     print(f"[INFO] Saved hub visualization to {out_path}")


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, required=True)
#     parser.add_argument("--out", type=str, required=True, help="Path to save PNG")
#     parser.add_argument("--top_k", type=int, default=20, help="Number of top hubs to visualize")
#     args = parser.parse_args()

#     # ---- load config ----
#     cfg = load_config(args.config)
#     data_dir = Path(cfg["data"]["data_dir"])

#     # ---- load datasets ----
#     chch = load_chch(data_dir / cfg["data"]["chch_file"], sep=cfg["data"].get("sep_chch", "\t"))
#     ddinter = load_ddinter(sorted(list(data_dir.glob(cfg["data"]["ddinter_shards_glob"]))))

#     decagon = load_decagon(data_dir / cfg["data"]["decagon_file"])
#     pos_all = merge_sources(chch, ddinter, decagon)

#     # ---- drug index ----
#     drugs = pd.unique(pd.concat([pos_all["drug_u"], pos_all["drug_v"]])).astype(str).tolist()
#     drug2id = {d: i for i, d in enumerate(drugs)}
#     id2drug = {i: d for d, i in drug2id.items()}

#     # ---- load priors ----
#     atc_map = load_atc_map(data_dir)
#     cyp_df = load_cyp_table(cfg["data"]["drug_cyp_file"])

#     # ---- build relation graph ----
#     graph_data = build_relation_graph(
#         drugs=drugs,
#         atc_map=atc_map,
#         cyp_df=cyp_df,
#         ddi_edges=pos_all[["drug_u", "drug_v"]].values,
#     )

#     edge_index = graph_data["edge_index"]
#     edge_type = graph_data["edge_type"]

#     print("Edge_index shape:", edge_index.shape)
#     print("Unique edge types (raw):", np.unique(edge_type.cpu().numpy()))

#     # ---- remap edge types to 0..num_rels-1 ----
#     raw_types = np.unique(edge_type.cpu().numpy())
#     type_map = {t: i for i, t in enumerate(raw_types)}
#     remapped = torch.tensor([type_map[t.item()] for t in edge_type], dtype=torch.long)
#     graph_data["edge_type"] = remapped

#     print(f"[INFO] Remapped edge types {raw_types} -> 0..{len(raw_types)-1}")
#     print("Unique edge types (after remap):", np.unique(graph_data["edge_type"].cpu().numpy()))
#     num_rels = len(raw_types)
#     print(f"Num relations detected: {num_rels}")

#     # ---- plot hubs ----
#     plot_top_hubs(graph_data, drug2id, id2drug, args.out, top_k=args.top_k)


# if __name__ == "__main__":
#     main()

# -----------------------------------------------------------------------------------------------------------------------------#
# src/scripts/visualize_rgcn.py
# Top 10 K
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import networkx as nx

from pathlib import Path
from src.utils.io import load_config
from src.data.ingest import load_chch, load_ddinter, load_decagon, merge_sources
from src.features.priors import load_atc_map, load_cyp_table
from src.features.graph_relations import build_relation_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out", type=str, required=True, help="Path to save PNG")
    args = parser.parse_args()

    # ---- load config ----
    cfg = load_config(args.config)
    data_dir = Path(cfg["data"]["data_dir"])

    # ---- load datasets ----
    chch = load_chch(data_dir / cfg["data"]["chch_file"], sep=cfg["data"].get("sep_chch", "\t"))
    ddinter = load_ddinter(sorted(list(data_dir.glob(cfg["data"]["ddinter_shards_glob"]))))

    decagon = load_decagon(data_dir / cfg["data"]["decagon_file"])
    pos_all = merge_sources(chch, ddinter, decagon)

    # ---- drug index ----
    drugs = pd.unique(pd.concat([pos_all["drug_u"], pos_all["drug_v"]])).astype(str).tolist()
    drug2id = {d: i for i, d in enumerate(drugs)}

    # ---- load priors ----
    atc_map = load_atc_map(data_dir)
    cyp_df = load_cyp_table(cfg["data"]["drug_cyp_file"])

    # ---- build relation graph ----
    graph_data = build_relation_graph(
        drugs=drugs,
        atc_map=atc_map,
        cyp_df=cyp_df,
        ddi_edges=pos_all[["drug_u", "drug_v"]].values,
    )

    edge_index = graph_data["edge_index"]
    edge_type = graph_data["edge_type"]

    # ---- remap edge types ----
    raw_types = np.unique(edge_type.cpu().numpy())
    type_map = {t: i for i, t in enumerate(raw_types)}
    remapped = torch.tensor([type_map[t.item()] for t in edge_type], dtype=torch.long)
    graph_data["edge_type"] = remapped
    num_rels = len(raw_types)

    # ---- build networkx graph ----
    G = nx.Graph()
    G.add_nodes_from(range(len(drugs)))

    edge_index_np = edge_index.cpu().numpy()
    edge_type_np = graph_data["edge_type"].cpu().numpy()

    for i in range(edge_index_np.shape[1]):
        u, v = edge_index_np[:, i]
        G.add_edge(u, v, rel=edge_type_np[i])

    # ---- find top-10 hubs ----
    degree_dict = dict(G.degree())
    top_hubs = sorted(degree_dict, key=degree_dict.get, reverse=True)[:10]

    H = nx.Graph()
    for hub in top_hubs:
        for nbr in G.neighbors(hub):
            H.add_edge(hub, nbr, rel=G[hub][nbr]["rel"])

    pos = nx.spring_layout(H, seed=42, k=0.6)

    # ---- draw ----
    plt.figure(figsize=(12, 12))
    ax = plt.gca()

    # color hubs differently
    hub_nodes = set(top_hubs)
    neighbor_nodes = set(H.nodes()) - hub_nodes

    nx.draw_networkx_nodes(H, pos, nodelist=hub_nodes, node_size=60, node_color="red", alpha=0.8, ax=ax, label="Hub drugs")
    nx.draw_networkx_nodes(H, pos, nodelist=neighbor_nodes, node_size=20, node_color="blue", alpha=0.6, ax=ax, label="Neighbors")

    # draw edges by type
    style_map = {0: "solid", 1: "dashed", 2: "dotted", 3: "dashdot",
                 4: "solid", 5: "dashed", 6: "dotted"}

    for u, v, d in H.edges(data=True):
        style = style_map.get(d["rel"], "solid")
        nx.draw_networkx_edges(H, pos, edgelist=[(u, v)], style=style, alpha=0.3, ax=ax)

    # label hubs
    labels = {hub: drugs[hub] for hub in top_hubs}
    nx.draw_networkx_labels(H, pos, labels, font_size=8, font_color="black", ax=ax)

    # add legend manually
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="black", lw=1, linestyle="solid", label="DDI edges"),
        Line2D([0], [0], color="black", lw=1, linestyle="dashed", label="ATC edges"),
        Line2D([0], [0], color="black", lw=1, linestyle="dotted", label="CYP edges"),
    ]
    ax.legend(handles=legend_elems, loc="upper right")

    plt.title("Top-10 Drug Hubs and Neighbors", fontsize=14)
    plt.axis("off")
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Saved visualization to {args.out}")


if __name__ == "__main__":
    main()


     