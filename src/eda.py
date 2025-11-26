from __future__ import annotations
import argparse
from collections import defaultdict
from typing import Dict
import pandas as pd
import networkx as nx

"""
MM-GRAPH–style EDA for MovieLens(+TMDB)

- Nodes: movies with at least one co-rating edge
- Edges: movie–movie links (two movies rated by ≥ min_coreviews of the same users)
- Average Degree: 2E/N
- Average CC: average clustering coefficient (undirected, unweighted)
- Average RA: mean Resource Allocation index over existing edges
- Transitivity: global clustering coefficient
- Edge Homophily: fraction of edges whose endpoints share the same *primary* genre

Inputs:
  --movie_csv   CSV with at least columns: movieId, ml_genres (pipe-separated)
  --ratings_csv CSV with at least columns: userId, movieId
"""

"""
===========================================
What the Script Extracts (and How)
===========================================

Nodes:
    Number of movies that remain after projecting user→movie ratings 
    into a movie–movie graph (movies with at least one co-rating edge).

Edges:
    Number of undirected links between movies.
    Two movies are linked if ≥ min_coreviews users rated both (default 1).
    Edge weight counts shared users (used only to prune).

Average Degree:
    2E / N, where N is the number of nodes and E is the number of edges.

Average CC (Average Clustering Coefficient):
    Mean of local clustering over all movie nodes — measures how triangle-rich 
    the neighborhoods are.

Average RA (Resource Allocation):
    For each existing edge (u, v),
        RA(u, v) = Σ_{z ∈ Γ(u) ∩ Γ(v)} 1 / deg(z)
    The script averages this value over all edges, mirroring the
    “Average RA” column in the MM-GRAPH paper.

Transitivity:
    Global clustering coefficient — the ratio of closed triplets 
    (triangles) to all triplets in the graph.

Edge Homophily:
    Fraction of edges whose endpoints share the same primary genre label.
    The script sets a movie’s label to the first token in ml_genres 
    (e.g., "Adventure|Animation|..." → "Adventure").
    Missing or empty genres default to "Unknown".

Tip:
    Increase --min_coreviews (e.g., 2 or 3) to denoise co-rating edges 
    for very large datasets. This typically lowers Nodes/Edges counts 
    but increases clustering and homophily stability.
"""


def primary_genre(genres: str) -> str:
    """First token in pipe-separated ml_genres; 'Unknown' if missing."""
    if not isinstance(genres, str) or not genres.strip():
        return "Unknown"
    tokens = [g.strip() for g in genres.split("|") if g.strip()]
    return tokens[0] if tokens else "Unknown"


def build_movie_projection(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    min_coreviews: int = 1
) -> nx.Graph:
    """
    Build an undirected movie–movie projection graph from user→movie ratings.

    Edge weight = number of distinct users who rated both movies.
    Edges with weight < min_coreviews are dropped.
    Isolated movies (after pruning) are removed.
    """
    # Keep only rated movies
    rated_movie_ids = set(ratings["movieId"].unique())
    movies = movies[movies["movieId"].isin(rated_movie_ids)].copy()

    # Label for homophily (primary genre)
    if "ml_genres" in movies.columns:
        movies["label"] = movies["ml_genres"].apply(primary_genre)
    else:
        movies["label"] = "Unknown"

    # Prepare node attributes
    node_attrs: Dict[int, dict] = movies.set_index("movieId").to_dict(orient="index")

    # Build user → list[movieId]
    user2movies = defaultdict(list)
    for row in ratings.itertuples(index=False):
        user2movies[int(row.userId)].append(int(row.movieId))

    # Create graph
    G = nx.Graph()
    for mid, attrs in node_attrs.items():
        G.add_node(int(mid), **attrs)

    # Add co-rating edges
    for movie_list in user2movies.values():
        # de-duplicate per user to avoid repeated pairs from multiple ratings
        uniq = list(dict.fromkeys(movie_list))
        n = len(uniq)
        for i in range(n):
            mi = uniq[i]
            if mi not in G:
                continue
            for j in range(i + 1, n):
                mj = uniq[j]
                if mj not in G:
                    continue
                if G.has_edge(mi, mj):
                    G[mi][mj]["weight"] += 1
                else:
                    G.add_edge(mi, mj, weight=1)

    # Prune weak edges
    if min_coreviews > 1:
        to_remove = [(u, v) for u, v, d in G.edges(data=True) if d.get("weight", 0) < min_coreviews]
        G.remove_edges_from(to_remove)

    # Remove isolates
    isolates = list(nx.isolates(G))
    if isolates:
        G.remove_nodes_from(isolates)

    return G


def average_resource_allocation(G: nx.Graph) -> float:
    """
    Average Resource Allocation (RA) index across existing edges.
    RA(u,v) = sum_{z in Γ(u)∩Γ(v)} 1/deg(z)
    """
    m = G.number_of_edges()
    if m == 0:
        return 0.0
    total = 0.0
    for u, v in G.edges():
        nu, nv = set(G.neighbors(u)), set(G.neighbors(v))
        ra = 0.0
        for z in nu & nv:
            degz = G.degree(z)
            if degz > 0:
                ra += 1.0 / degz
        total += ra
    return total / m


def edge_homophily(G: nx.Graph, label_key: str = "label") -> float:
    """
    Fraction of edges whose endpoints share the same categorical label.
    Here label = primary movie genre.
    """
    m = G.number_of_edges()
    if m == 0:
        return 0.0
    same = 0
    for u, v in G.edges():
        if G.nodes[u].get(label_key, "Unknown") == G.nodes[v].get(label_key, "Unknown"):
            same += 1
    return same / m


def compute_stats(G: nx.Graph) -> dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    avg_deg = (2.0 * m / n) if n else 0.0
    avg_cc = nx.average_clustering(G) if n else 0.0
    trans = nx.transitivity(G) if n else 0.0
    avg_ra = average_resource_allocation(G)
    eh = edge_homophily(G, "label")
    return {
        "Nodes": n,
        "Edges": m,
        "Average Degree": round(avg_deg, 4),
        "Average CC": round(avg_cc, 4),
        "Average RA": round(avg_ra, 4),
        "Transitivity": round(trans, 4),
        "Edge Homophily": round(eh, 4),
    }


def run(movie_csv: str, ratings_csv: str, min_coreviews: int = 1, out_csv: str | None = None):
    # Load
    movies = pd.read_csv(movie_csv)
    ratings = pd.read_csv(ratings_csv)

    # Cast IDs defensively
    for col in ("movieId", "userId"):
        if col in ratings.columns:
            ratings[col] = pd.to_numeric(ratings[col], errors="coerce").astype("Int64")
    if "movieId" in movies.columns:
        movies["movieId"] = pd.to_numeric(movies["movieId"], errors="coerce").astype("Int64")

    # Drop bad rows
    ratings = ratings.dropna(subset=["movieId", "userId"])
    movies = movies.dropna(subset=["movieId"])

    # Build projection and compute stats
    G = build_movie_projection(movies, ratings, min_coreviews=min_coreviews)
    stats = compute_stats(G)

    # Emit
    df = pd.DataFrame([stats])
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description="Compute MM-GRAPH–style EDA table for MovieLens(+TMDB).")
    ap.add_argument("--movie_csv", required=True, help="Path to movie metadata CSV (ml32m_tmdb_enriched_full.csv)")
    ap.add_argument("--ratings_csv", required=True, help="Path to ratings CSV")
    ap.add_argument("--min_coreviews", type=int, default=1, help="Min shared users to keep a movie–movie edge (default: 1)")
    ap.add_argument("--out_csv", default=None, help="Optional: path to write the result table as CSV")
    args = ap.parse_args()

    df = run(args.movie_csv, args.ratings_csv, args.min_coreviews, args.out_csv)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
