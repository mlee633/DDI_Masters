# def build_relation_graph(drugs, atc_map, cyp_df, ddi_edges):
#     drug2id = {d: i for i, d in enumerate(drugs)}
#     edge_index, edge_type = [], []
    
#     # ATC hierarchy edges
#     for d1, codes1 in atc_map.items():
#         if d1 not in drug2id:
#             continue
#         for d2, codes2 in atc_map.items():
#             if d2 not in drug2id:
#                 continue
#             # only connect if they share ATC prefix
#             for lvl in range(1, 5):
#                 if any(c1[:lvl] == c2[:lvl] for c1 in codes1 for c2 in codes2):
#                     edge_index.append([drug2id[d1], drug2id[d2]])
#                     edge_type.append(lvl)  # type id per level
    
#     # CYP edges
#     for enzyme in cyp_df.columns[1:]:
#         subs = cyp_df[cyp_df[enzyme] == 1]["drug_id"].tolist()
#         for i in range(len(subs)):
#             for j in range(i + 1, len(subs)):
#                 if subs[i] in drug2id and subs[j] in drug2id:
#                     edge_index.append([drug2id[subs[i]], drug2id[subs[j]]])
#                     edge_type.append(10)  # example code for CYP
    
#     # DDI edges
#     for u, v in ddi_edges:
#         if u in drug2id and v in drug2id:
#             edge_index.append([drug2id[u], drug2id[v]])
#             edge_type.append(20)

#     import torch
#     edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#     edge_type = torch.tensor(edge_type, dtype=torch.long)
#     x = torch.eye(len(drugs))  # identity features for each drug

#     return {"x": x, "edge_index": edge_index, "edge_type": edge_type}



# src/features/graph_relations.py

import torch
import numpy as np

def build_relation_graph(drugs, atc_map, cyp_df, ddi_edges):
    """
    Build a multi-relational graph for RGCN/RGAT with relation types:
      - DDI positive edges
      - ATC hierarchy edges (level 1–5)
      - CYP substrate/inhibitor/inducer relations
    """
    edge_index = []
    edge_type = []

    rel_id = 0
    rel_names = {}

    # 1. DDI positive edges
    for u, v in ddi_edges:
        if u in drugs and v in drugs:
            edge_index.append([drugs.index(u), drugs.index(v)])
            edge_type.append(rel_id)
            edge_index.append([drugs.index(v), drugs.index(u)])
            edge_type.append(rel_id)
    rel_names[rel_id] = "ddi_positive"
    rel_id += 1

    # 2. ATC relations
    for lvl in range(1, 6):
        for d1 in drugs:
            for d2 in drugs:
                if d1 == d2:
                    continue
                A = atc_map.get(d1, [])
                B = atc_map.get(d2, [])
                if any(a[:lvl] == b[:lvl] for a in A for b in B):
                    edge_index.append([drugs.index(d1), drugs.index(d2)])
                    edge_type.append(rel_id)
        rel_names[rel_id] = f"atc_level_{lvl}"
        rel_id += 1

    # 3. CYP relations
    if cyp_df is not None:
        for enzyme in cyp_df.columns[1:]:
            subs = cyp_df[cyp_df[enzyme] == 1]["drug_id"].tolist()
            for i in range(len(subs)):
                for j in range(i + 1, len(subs)):
                    if subs[i] in drugs and subs[j] in drugs:
                        edge_index.append([drugs.index(subs[i]), drugs.index(subs[j])])
                        edge_type.append(rel_id)
                        edge_index.append([drugs.index(subs[j]), drugs.index(subs[i])])
                        edge_type.append(rel_id)
            rel_names[rel_id] = f"{enzyme}"
            rel_id += 1

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long)

    # --- FIX: remap edge types to contiguous 0..N-1 ---
    unique_rels = np.unique(edge_type.cpu().numpy())
    rel_map = {old: new for new, old in enumerate(unique_rels)}
    edge_type = torch.tensor([rel_map[int(e)] for e in edge_type], dtype=torch.long)

    print(f"[INFO] Remapped edge types {unique_rels.tolist()} -> 0..{len(unique_rels)-1}")

    return {
        "x": torch.arange(len(drugs), dtype=torch.long),
        "edge_index": edge_index,
        "edge_type": edge_type,
        "num_rels": len(unique_rels),
        "rel_names": rel_names,
    }
