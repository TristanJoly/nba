"""
graph_builder.py – Build and export the team ↔ player ↔ coach relationship graph.

The graph is stored in two places:
  1. SQLite table *graph_edges* (persistent, queryable)
  2. In-memory NetworkX MultiDiGraph (for analysis and export)

Node types
----------
  team    – NBA franchise (node id = "team_{team_id}")
  player  – player (node id = "player_{player_id}")
  coach   – head coach (node id = "coach_{coach_id}")

Edge types
----------
  plays_for   player  → team  (per season)
  coaches     coach   → team  (per season)
  drafted_by  player  → team  (draft year)
  teammate    player  ↔ player (share same team × season; weight = seasons together)

Example usage
-------------
  from graph_builder import build_graph, export_gexf, analyse_graph
  G = build_graph(season="2023-24")
  export_gexf(G, "nba_graph_2023-24.gexf")
  analyse_graph(G)
"""

import logging
from collections import defaultdict
from typing import Optional

import networkx as nx
import pandas as pd

from database import get_connection, upsert_df
from config import CURRENT_SEASON

logger = logging.getLogger(__name__)


# ===========================================================================
# BUILD GRAPH
# ===========================================================================

def build_graph(
    season: Optional[str] = None,
    db_path: str = "nba_draft_analytics.db",
    include_teammates: bool = True,
) -> nx.MultiDiGraph:
    """
    Build a directed multigraph from the database.

    Parameters
    ----------
    season : str, optional
        If provided, only include edges for that season.
        If None, include all seasons (edges carry a 'season' attribute).
    include_teammates : bool
        Whether to add player ↔ player 'teammate' edges.
        Can be expensive for many seasons.

    Returns
    -------
    nx.MultiDiGraph
    """
    G = nx.MultiDiGraph()
    conn = get_connection(db_path)

    # ── Teams ─────────────────────────────────────────────────────────────────
    teams = conn.execute("SELECT * FROM teams").fetchall()
    for t in teams:
        nid = f"team_{t['team_id']}"
        G.add_node(nid, node_type="team", team_id=t["team_id"],
                   name=t["full_name"], abbreviation=t["abbreviation"],
                   conference=t["conference"], division=t["division"])

    # ── Players ───────────────────────────────────────────────────────────────
    players = conn.execute("SELECT * FROM players").fetchall()
    for p in players:
        nid = f"player_{p['player_id']}"
        G.add_node(nid, node_type="player", player_id=p["player_id"],
                   name=p["full_name"], position=p["position"],
                   height=p["height"], weight=p["weight"],
                   school=p["school"], country=p["country"],
                   draft_year=p["draft_year"], draft_number=p["draft_number"],
                   is_active=p["is_active"])

    # ── Coaches ───────────────────────────────────────────────────────────────
    season_filter = "WHERE season = ?" if season else ""
    params = (season,) if season else ()

    coaches = conn.execute(
        f"SELECT DISTINCT coach_id, team_id, first_name, last_name, coach_type, season "
        f"FROM coaches {season_filter}", params
    ).fetchall()

    for c in coaches:
        nid = f"coach_{c['coach_id']}"
        if nid not in G:
            G.add_node(nid, node_type="coach", coach_id=c["coach_id"],
                       name=f"{c['first_name']} {c['last_name']}",
                       coach_type=c["coach_type"])
        # coaches → team
        team_nid = f"team_{c['team_id']}"
        G.add_edge(nid, team_nid, edge_type="coaches", season=c["season"])

    # ── Roster edges: player → team ───────────────────────────────────────────
    roster_rows = conn.execute(
        f"SELECT player_id, team_id, season FROM team_roster {season_filter}", params
    ).fetchall()

    for r in roster_rows:
        p_nid = f"player_{r['player_id']}"
        t_nid = f"team_{r['team_id']}"
        if p_nid in G and t_nid in G:
            G.add_edge(p_nid, t_nid, edge_type="plays_for", season=r["season"], weight=1.0)

    # ── Draft edges: player → team ────────────────────────────────────────────
    draft_rows = conn.execute(
        "SELECT player_id, team_id, season FROM draft_history"
    ).fetchall()

    for d in draft_rows:
        p_nid = f"player_{d['player_id']}"
        t_nid = f"team_{d['team_id']}"
        if p_nid in G and t_nid in G:
            G.add_edge(p_nid, t_nid, edge_type="drafted_by", season=d["season"], weight=1.0)

    # ── Teammate edges (optional, expensive) ─────────────────────────────────
    if include_teammates:
        logger.info("Building teammate edges…")
        # Group players by (team, season)
        team_season_players: dict[tuple, list[int]] = defaultdict(list)
        for r in roster_rows:
            team_season_players[(r["team_id"], r["season"])].append(r["player_id"])

        teammate_weight: dict[tuple[int, int], float] = defaultdict(float)
        for players_list in team_season_players.values():
            for i, p1 in enumerate(players_list):
                for p2 in players_list[i + 1:]:
                    key = (min(p1, p2), max(p1, p2))
                    teammate_weight[key] += 1.0

        for (p1, p2), w in teammate_weight.items():
            n1, n2 = f"player_{p1}", f"player_{p2}"
            if n1 in G and n2 in G:
                G.add_edge(n1, n2, edge_type="teammate", weight=w)

    conn.close()
    logger.info(
        f"✅ Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
        f"(season={'all' if season is None else season})"
    )
    return G


# ===========================================================================
# PERSIST EDGES TO DB
# ===========================================================================

def persist_graph_edges(G: nx.MultiDiGraph, db_path: str = "nba_draft_analytics.db") -> None:
    """Write all edges from G into the *graph_edges* table."""
    rows = []
    for src, dst, data in G.edges(data=True):
        src_type = G.nodes[src].get("node_type", "unknown")
        dst_type = G.nodes[dst].get("node_type", "unknown")
        rows.append({
            "source_type":  src_type,
            "source_id":    G.nodes[src].get(f"{src_type}_id", 0),
            "source_name":  G.nodes[src].get("name", src),
            "target_type":  dst_type,
            "target_id":    G.nodes[dst].get(f"{dst_type}_id", 0),
            "target_name":  G.nodes[dst].get("name", dst),
            "edge_type":    data.get("edge_type", ""),
            "season":       data.get("season", ""),
            "weight":       data.get("weight", 1.0),
        })

    if rows:
        df = pd.DataFrame(rows)
        conn = get_connection(db_path)
        upsert_df(df, "graph_edges", conn)
        conn.close()
        logger.info(f"✅ {len(rows):,} graph edges persisted to DB.")


# ===========================================================================
# EXPORTS
# ===========================================================================

def _without_none(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """GEXF/GraphML writers crash on None attribute values – drop them on a copy."""
    H = G.copy()
    for _, data in H.nodes(data=True):
        for k in [k for k, v in data.items() if v is None]:
            del data[k]
    for _, _, data in H.edges(data=True):
        for k in [k for k, v in data.items() if v is None]:
            del data[k]
    return H


def export_gexf(G: nx.MultiDiGraph, path: str = "nba_graph.gexf") -> None:
    """Export graph as GEXF (openable in Gephi or yEd)."""
    nx.write_gexf(_without_none(G), path)
    logger.info(f"Graph exported → {path}")


def export_graphml(G: nx.MultiDiGraph, path: str = "nba_graph.graphml") -> None:
    """Export graph as GraphML."""
    nx.write_graphml(_without_none(G), path)
    logger.info(f"Graph exported → {path}")


def export_edgelist(G: nx.MultiDiGraph, path: str = "nba_edges.csv") -> None:
    """Export edge list as CSV for use in Gephi / Cytoscape / d3."""
    rows = []
    for src, dst, data in G.edges(data=True):
        rows.append({
            "source": src,
            "target": dst,
            **{k: v for k, v in data.items()},
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info(f"Edge list exported → {path}")


# ===========================================================================
# ANALYSIS HELPERS
# ===========================================================================

def analyse_graph(G: nx.MultiDiGraph) -> dict:
    """
    Print and return basic graph statistics.

    Returns
    -------
    dict with keys: n_nodes, n_edges, n_teams, n_players, n_coaches,
                    most_connected_players, most_connected_teams
    """
    node_types = {t: 0 for t in ("team", "player", "coach")}
    for _, data in G.nodes(data=True):
        nt = data.get("node_type", "unknown")
        node_types[nt] = node_types.get(nt, 0) + 1

    # Degree centrality for players only
    player_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "player"]
    player_degree = {n: G.degree(n) for n in player_nodes}
    top_players = sorted(player_degree.items(), key=lambda x: x[1], reverse=True)[:10]

    team_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "team"]
    team_degree = {n: G.degree(n) for n in team_nodes}
    top_teams = sorted(team_degree.items(), key=lambda x: x[1], reverse=True)[:5]

    stats = {
        "n_nodes":   G.number_of_nodes(),
        "n_edges":   G.number_of_edges(),
        "n_teams":   node_types.get("team", 0),
        "n_players": node_types.get("player", 0),
        "n_coaches": node_types.get("coach", 0),
        "most_connected_players": [
            (G.nodes[n].get("name", n), deg) for n, deg in top_players
        ],
        "most_connected_teams": [
            (G.nodes[n].get("name", n), deg) for n, deg in top_teams
        ],
    }

    print("\n📊 Graph Analysis")
    print(f"   Nodes  : {stats['n_nodes']:,}  "
          f"(teams={stats['n_teams']}, players={stats['n_players']}, coaches={stats['n_coaches']})")
    print(f"   Edges  : {stats['n_edges']:,}")
    print("\n   Most-connected players (by degree):")
    for name, deg in stats["most_connected_players"]:
        print(f"     {name:<30s} degree={deg}")
    print("\n   Most-connected teams (by degree):")
    for name, deg in stats["most_connected_teams"]:
        print(f"     {name:<30s} degree={deg}")

    return stats


def get_player_subgraph(
    G: nx.MultiDiGraph, player_id: int, depth: int = 1
) -> nx.MultiDiGraph:
    """
    Extract the ego-graph centred on *player_id* up to *depth* hops.
    Useful for exploring a single prospect's NBA connections.
    """
    node = f"player_{player_id}"
    if node not in G:
        logger.warning(f"Player {player_id} not in graph.")
        return nx.MultiDiGraph()
    ego = nx.ego_graph(G.to_undirected(as_view=True), node, radius=depth)
    return G.subgraph(ego.nodes()).copy()
