import pandas as pd
import networkx as nx


def build_graph_from_pos(pos_df):
    """Build a NetworkX graph from positive interaction pairs."""
    G = nx.Graph()
    for _, r in pos_df.iterrows():
        G.add_edge(r["drug_u"], r["drug_v"])
    return G


def pair_struct_features(G, pairs_df):
    """
    Compute simple structural features for drug pairs.
    If a drug is not present in the training graph, features default to 0.
    """
    deg = dict(G.degree())
    safe_deg = lambda x: deg.get(x, 0)

    u = pairs_df["drug_u"].tolist()
    v = pairs_df["drug_v"].tolist()

    feats = {
        "deg_u": [safe_deg(x) for x in u],
        "deg_v": [safe_deg(x) for x in v],
    }

    # Only evaluate similarity scores for pairs where both nodes are in G
    valid_pairs = [(a, b) for a, b in zip(u, v) if a in G and b in G]

    jacc, aa, ra = {}, {}, {}
    if valid_pairs:
        jacc.update({(a, b): s for a, b, s in nx.jaccard_coefficient(G, valid_pairs)})
        aa.update({(a, b): s for a, b, s in nx.adamic_adar_index(G, valid_pairs)})
        ra.update({(a, b): s for a, b, s in nx.resource_allocation_index(G, valid_pairs)})

    # Create keys with consistent ordering
    key = [(min(a, b), max(a, b)) for a, b in zip(u, v)]
    feats["jaccard"] = [jacc.get(k, 0.0) for k in key]
    feats["adamic_adar"] = [aa.get(k, 0.0) for k in key]
    feats["res_alloc"] = [ra.get(k, 0.0) for k in key]

    return pd.DataFrame(feats)
