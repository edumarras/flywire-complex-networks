from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "connections_princeton.csv.gz"
OUT_DIR = ROOT / "reports" / "f1"/ "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "flywire_subgraph_visualization.png"


def minmax_scale(values, new_min, new_max):
    values = np.asarray(values, dtype=float)
    vmin = values.min()
    vmax = values.max()
    if np.isclose(vmin, vmax):
        return np.full_like(values, (new_min + new_max) / 2.0)
    return new_min + (values - vmin) * (new_max - new_min) / (vmax - vmin)


def main():
    print("Lendo arestas...")
    df = pd.read_csv(
        DATA_PATH,
        compression="gzip",
        usecols=["pre_root_id", "post_root_id", "syn_count"],
        dtype={
            "pre_root_id": "int64",
            "post_root_id": "int64",
            "syn_count": "int32",
        },
    )

    # Remove auto-loops, se existirem
    df = df[df["pre_root_id"] != df["post_root_id"]].copy()

    # Caso existam múltiplas linhas para o mesmo par, agrega
    df = (
        df.groupby(["pre_root_id", "post_root_id"], as_index=False)["syn_count"]
        .sum()
    )

    print("Calculando graus...")
    out_deg = df.groupby("pre_root_id").size().rename("out_deg")
    in_deg = df.groupby("post_root_id").size().rename("in_deg")

    deg = pd.concat([out_deg, in_deg], axis=1).fillna(0)
    deg["total_deg"] = deg["out_deg"] + deg["in_deg"]

    # Escolha dos hubs
    N_HUBS = 25
    N_NEIGHBORS_PER_HUB = 15

    top_hubs = deg.nlargest(N_HUBS, "total_deg").index.tolist()
    selected_nodes = set(top_hubs)

    print("Selecionando vizinhos fortes dos hubs...")
    for hub in top_hubs:
        outgoing = (
            df.loc[df["pre_root_id"] == hub, ["post_root_id", "syn_count"]]
            .rename(columns={"post_root_id": "neighbor"})
        )
        incoming = (
            df.loc[df["post_root_id"] == hub, ["pre_root_id", "syn_count"]]
            .rename(columns={"pre_root_id": "neighbor"})
        )

        neigh = pd.concat([outgoing, incoming], ignore_index=True)
        if len(neigh) == 0:
            continue

        neigh = (
            neigh.groupby("neighbor", as_index=False)["syn_count"]
            .sum()
            .sort_values("syn_count", ascending=False)
            .head(N_NEIGHBORS_PER_HUB)
        )

        selected_nodes.update(neigh["neighbor"].tolist())

    print(f"Nós selecionados: {len(selected_nodes)}")

    print("Construindo subgrafo...")
    sub_df = df[
        df["pre_root_id"].isin(selected_nodes) &
        df["post_root_id"].isin(selected_nodes)
    ].copy()

    # Visualização em versão não direcionada para ficar mais legível
    G = nx.Graph()
    for row in sub_df.itertuples(index=False):
        u = row.pre_root_id
        v = row.post_root_id
        w = int(row.syn_count)

        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)

    # Foca na maior componente conexa do subgrafo
    if G.number_of_nodes() == 0:
        raise RuntimeError("Subgrafo vazio. Ajuste os parâmetros de seleção.")

    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

    print(f"Subgrafo final: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas")

    print("Detectando comunidades...")
    communities = list(nx.community.greedy_modularity_communities(G, weight="weight"))
    community_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            community_map[node] = i

    node_colors = [community_map[n] for n in G.nodes()]
    node_degrees = np.array([G.degree(n) for n in G.nodes()], dtype=float)
    node_sizes = minmax_scale(node_degrees, 120, 1800)

    edge_weights = np.array([G[u][v]["weight"] for u, v in G.edges()], dtype=float)
    edge_widths = minmax_scale(edge_weights, 0.4, 3.5)

    print("Calculando layout...")
    pos = nx.spring_layout(
        G,
        weight="weight",
        seed=42,
        k=1.1 / np.sqrt(max(G.number_of_nodes(), 2)),
        iterations=200,
    )

    print("Gerando SVG...")

    from html import escape

    WIDTH = 1600
    HEIGHT = 1200
    MARGIN = 80

    # Normaliza as posições do spring_layout para coordenadas da imagem
    x_vals = np.array([pos[n][0] for n in G.nodes()])
    y_vals = np.array([pos[n][1] for n in G.nodes()])

    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()

    def transform_xy(x, y):
        px = MARGIN + (x - x_min) / (x_max - x_min) * (WIDTH - 2 * MARGIN)

        # SVG cresce para baixo, então invertemos Y
        py = HEIGHT - (
            MARGIN + (y - y_min) / (y_max - y_min) * (HEIGHT - 2 * MARGIN)
        )

        return px, py


    # Tamanho dos nós
    node_degree_map = dict(G.degree())

    degrees = np.array(
        [node_degree_map[n] for n in G.nodes()],
        dtype=float,
    )

    degree_min = degrees.min()
    degree_max = degrees.max()

    def node_radius(node):
        d = node_degree_map[node]

        if degree_max == degree_min:
            return 6

        normalized = (d - degree_min) / (degree_max - degree_min)

        return 3 + normalized * 15


    # Cor por comunidade
    n_communities = max(community_map.values()) + 1

    def community_color(node):
        c = community_map[node]

        hue = int((360 * c) / max(n_communities, 1))

        return f"hsl({hue}, 65%, 50%)"


    svg_path = OUT_DIR / "flywire_subgraph_visualization.svg"

    parts = []

    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}">'
    )

    parts.append("""
    <rect width="100%" height="100%" fill="white"/>
    """)

    # Título
    parts.append(
        f'''
        <text
            x="{WIDTH / 2}"
            y="40"
            text-anchor="middle"
            font-family="Arial"
            font-size="24"
            font-weight="bold">
            FlyWire Connectome — subgrafo centrado em hubs
        </text>
        '''
    )

    parts.append(
        f'''
        <text
            x="{WIDTH / 2}"
            y="68"
            text-anchor="middle"
            font-family="Arial"
            font-size="16">
            tamanho do nó = grau; cor = comunidade
        </text>
        '''
    )


    # -------------------------
    # ARESTAS
    # -------------------------

    for u, v in G.edges():
        x1, y1 = transform_xy(*pos[u])
        x2, y2 = transform_xy(*pos[v])

        parts.append(
            f'''
            <line
                x1="{x1:.2f}"
                y1="{y1:.2f}"
                x2="{x2:.2f}"
                y2="{y2:.2f}"
                stroke="black"
                stroke-width="0.7"
                stroke-opacity="0.12"/>
            '''
        )


    # -------------------------
    # NÓS
    # -------------------------

    for node in G.nodes():
        x, y = transform_xy(*pos[node])

        radius = node_radius(node)
        color = community_color(node)

        parts.append(
            f'''
            <circle
                cx="{x:.2f}"
                cy="{y:.2f}"
                r="{radius:.2f}"
                fill="{color}"
                fill-opacity="0.90"
                stroke="black"
                stroke-width="0.4"
                stroke-opacity="0.5"/>
            '''
        )


    # -------------------------
    # LABELS DOS MAIORES HUBS
    # -------------------------

    top_label_nodes = sorted(
        G.degree,
        key=lambda x: x[1],
        reverse=True,
    )[:12]

    for node, degree in top_label_nodes:
        x, y = transform_xy(*pos[node])

        label = escape(str(node))

        parts.append(
            f'''
            <text
                x="{x + 8:.2f}"
                y="{y - 8:.2f}"
                font-family="Arial"
                font-size="12"
                fill="black">
                {label}
            </text>
            '''
        )


    parts.append("</svg>")

    svg_path.write_text(
        "\n".join(parts),
        encoding="utf-8",
    )

    print(f"SVG salvo em:")
    print(svg_path.resolve())

    print(f"Existe? {svg_path.exists()}")

    if svg_path.exists():
        print(
            f"Tamanho: {svg_path.stat().st_size / 1024:.1f} KB"
        )


if __name__ == "__main__":
    main()