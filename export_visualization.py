from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer


OUTPUT_PATH = Path("embedding-vectors-visualization.html")
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


def to_numpy(embeddings):
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().cpu().numpy()
    return np.asarray(embeddings, dtype=float)


def reduce_to_3d(vectors):
    vectors = np.asarray(vectors, dtype=float)
    centered = vectors - vectors.mean(axis=0, keepdims=True)

    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    component_count = min(3, vh.shape[0])
    reduced = centered @ vh[:component_count].T

    if reduced.shape[1] < 3:
        reduced = np.pad(reduced, ((0, 0), (0, 3 - reduced.shape[1])), mode="constant")

    return reduced[:, :3]


def add_plotly_group(fig, name, points, labels, descriptions, color, symbol):
    line_x, line_y, line_z = [], [], []
    for x, y, z in points:
        line_x.extend([0, x, None])
        line_y.extend([0, y, None])
        line_z.extend([0, z, None])

    fig.add_trace(
        go.Scatter3d(
            x=line_x,
            y=line_y,
            z=line_z,
            mode="lines",
            line={"color": color, "width": 4},
            name=f"{name} vectors",
            legendgroup=name,
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker={"size": 6, "color": color, "symbol": symbol},
            name=name,
            legendgroup=name,
            customdata=np.array(descriptions, dtype=object),
            hovertemplate="<b>%{text}</b><br>%{customdata}<extra></extra>",
        )
    )


def main():
    model = SentenceTransformer(MODEL_NAME)

    documents = [
        "The Eiffel Tower is located in Paris and was completed in 1889.",
        "Photosynthesis allows plants to convert sunlight, water, and carbon dioxide into energy.",
        "Python is a popular programming language used for web development, data science, and automation.",
        "Mount Everest is the highest mountain above sea level on Earth.",
        "Octopuses are highly intelligent marine animals with eight arms and remarkable camouflage abilities.",
        "Basketball is a team sport in which players score points by shooting a ball through a hoop.",
    ]

    queries = [
        "Where is the Eiffel Tower located?",
        "How do plants make food from sunlight?",
        "What is the tallest mountain in the world?",
        "Tell me about the Python programming language.",
        "How do you bake sourdough bread?",
        "What causes thunderstorms to form?",
    ]

    multilingual_queries = [
        "Où se trouve la tour Eiffel ?",
        "¿Cómo producen energía las plantas con la luz solar?",
        "Qual è la montagna più alta del mondo?",
        "Pythonプログラミング言語について教えてください。",
        "Comment faire du pain au levain ?",
        "¿Qué causa la formación de tormentas eléctricas?",
    ]

    document_embeddings = model.encode(documents)
    query_embeddings = model.encode(queries, prompt_name="query")
    multilingual_query_embeddings = model.encode(multilingual_queries, prompt_name="query")

    all_vectors = np.vstack([
        to_numpy(document_embeddings),
        to_numpy(query_embeddings),
        to_numpy(multilingual_query_embeddings),
    ])
    projected_vectors = reduce_to_3d(all_vectors)

    num_documents = len(documents)
    num_queries = len(queries)
    projected_documents = projected_vectors[:num_documents]
    projected_queries = projected_vectors[num_documents:num_documents + num_queries]
    projected_multilingual_queries = projected_vectors[num_documents + num_queries:]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode="markers+text",
            text=["Origin"],
            textposition="top center",
            marker={"size": 5, "color": "black", "symbol": "x"},
            name="Origin",
            hovertemplate="<b>Origin</b><extra></extra>",
        )
    )

    add_plotly_group(
        fig,
        "Documents",
        projected_documents,
        [f"D{index}" for index in range(1, len(projected_documents) + 1)],
        documents,
        "#1f77b4",
        "circle",
    )
    add_plotly_group(
        fig,
        "English queries",
        projected_queries,
        [f"Q{index}" for index in range(1, len(projected_queries) + 1)],
        queries,
        "#d62728",
        "diamond",
    )
    add_plotly_group(
        fig,
        "Multilingual queries",
        projected_multilingual_queries,
        [f"M{index}" for index in range(1, len(projected_multilingual_queries) + 1)],
        multilingual_queries,
        "#2ca02c",
        "square",
    )

    fig.update_layout(
        title="Interactive 3D Visualization of Document and Query Embeddings",
        legend={"groupclick": "togglegroup"},
        scene={
            "xaxis_title": "Principal component 1",
            "yaxis_title": "Principal component 2",
            "zaxis_title": "Principal component 3",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )

    fig.write_html(OUTPUT_PATH, include_plotlyjs=True, full_html=True)
    print(f"Saved interactive HTML visualization to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()

