"""
sod_graph.py
Segregation of Duties (SoD) conflict graph using NetworkX.
Detects approver-submitter collusion patterns — the #1 SOX audit concern in SAP/Oracle ERP.
"""

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def build_sod_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed graph where:
    - Nodes = employees (submitters and approvers)
    - Edges = transaction relationships (submitter → approver)
    - Edge weight = number of transactions
    Self-loops indicate SoD violations (self-approval).
    """
    G = nx.DiGraph()

    for _, row in df.iterrows():
        src = row["submitter_id"]
        tgt = row["approver_id"]
        if G.has_edge(src, tgt):
            G[src][tgt]["weight"] += 1
            G[src][tgt]["total_amount"] += row["amount"]
        else:
            G.add_edge(src, tgt, weight=1, total_amount=row["amount"])

    return G


def detect_violations(G: nx.DiGraph, min_transactions: int = 3) -> pd.DataFrame:
    """
    Detect SoD violations:
    1. Self-loops (self-approval)
    2. Mutual approval (A approves B AND B approves A — collusion ring)
    3. High-frequency single-approver pairs
    """
    violations = []

    # Self-approval
    for node in G.nodes():
        if G.has_edge(node, node):
            data = G[node][node]
            violations.append({
                "violation_type": "SELF_APPROVAL",
                "submitter": node,
                "approver": node,
                "transaction_count": data["weight"],
                "total_amount": round(data["total_amount"], 2),
                "risk": "CRITICAL"
            })

    # Mutual approval (collusion rings)
    checked = set()
    for u, v in G.edges():
        if u != v and (v, u) not in checked and G.has_edge(v, u):
            checked.add((u, v))
            uv = G[u][v]
            vu = G[v][u]
            if uv["weight"] >= min_transactions and vu["weight"] >= min_transactions:
                violations.append({
                    "violation_type": "MUTUAL_APPROVAL_RING",
                    "submitter": f"{u} <-> {v}",
                    "approver": "BIDIRECTIONAL",
                    "transaction_count": uv["weight"] + vu["weight"],
                    "total_amount": round(uv["total_amount"] + vu["total_amount"], 2),
                    "risk": "HIGH"
                })

    # High-frequency pairs (concentration risk)
    for u, v, data in G.edges(data=True):
        if u != v and data["weight"] >= 50:
            violations.append({
                "violation_type": "HIGH_CONCENTRATION",
                "submitter": u,
                "approver": v,
                "transaction_count": data["weight"],
                "total_amount": round(data["total_amount"], 2),
                "risk": "MEDIUM"
            })

    return pd.DataFrame(violations).sort_values("risk", ascending=True).reset_index(drop=True)


def plot_sod_graph(G: nx.DiGraph, violations_df: pd.DataFrame, output_path: str = "reports/sod_graph.png"):
    """Visualize SoD conflict graph with violation highlights."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Only plot the violation subgraph for clarity
    violation_nodes = set()
    for _, row in violations_df.iterrows():
        if "<->" in str(row["submitter"]):
            parts = row["submitter"].split(" <-> ")
            violation_nodes.update(parts)
        else:
            violation_nodes.add(row["submitter"])
            violation_nodes.add(row["approver"])

    subG = G.subgraph(list(violation_nodes)[:50])  # cap at 50 nodes for readability

    pos = nx.spring_layout(subG, seed=42, k=2)

    # Color by violation type
    self_approvers = {r["submitter"] for _, r in violations_df.iterrows() if r["violation_type"] == "SELF_APPROVAL"}
    node_colors = ["#E24B4A" if n in self_approvers else "#378ADD" for n in subG.nodes()]

    nx.draw_networkx_nodes(subG, pos, node_color=node_colors, node_size=600, alpha=0.85, ax=ax)
    nx.draw_networkx_labels(subG, pos, font_size=7, font_color="white", font_weight="bold", ax=ax)
    nx.draw_networkx_edges(subG, pos, edge_color="#888780", arrows=True,
                           arrowsize=15, width=0.8, alpha=0.6, ax=ax)

    ax.set_title("SoD Conflict Graph — Red: Self-Approval Violations | Blue: Concentration Risk",
                 fontsize=12, pad=12)
    ax.axis("off")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"SoD graph saved to {output_path}")
    return fig


if __name__ == "__main__":
    import sys
    df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data/processed/transactions.csv")
    G = build_sod_graph(df)
    violations = detect_violations(G)
    print(f"\nDetected {len(violations)} SoD violations:")
    print(violations.to_string(index=False))
    plot_sod_graph(G, violations)
