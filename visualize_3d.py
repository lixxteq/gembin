import gradio as gr
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image
import io
from itertools import combinations

def compute_similarities(names, vectors):
    """Compute cosine similarity and Euclidean distance for each pair."""
    results = []
    for (i, name1), (j, name2) in combinations(enumerate(names), 2):
        v1 = vectors[i]
        v2 = vectors[j]
        cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        euclidean_dist = np.linalg.norm(v1 - v2)
        results.append({
            "Vector 1": name1,
            "Vector 2": name2,
            "Cosine Similarity": cos_sim,
            "Euclidean Distance": euclidean_dist
        })
    return pd.DataFrame(results)

def plot_embeddings(file):
    # load TSV
    df = pd.read_csv(file.name, sep="\t")
    names = df.iloc[:, 0].tolist()
    vectors = df.iloc[:, 1:].values.astype(float)
    n_samples, n_features = vectors.shape

    # info table: name and all dimension values
    info_df = df.copy()
    info_df.columns = ["Name"] + [f"Dim {i+1}" for i in range(n_features)]

    # determine number of components for PCA
    n_components = min(3, n_samples, n_features)
    pca = PCA(n_components=n_components)
    vectors_reduced = pca.fit_transform(vectors)

    # plot
    plt.figure(figsize=(8, 6))
    if n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        ax = plt.subplot(111, projection='3d')
        for i, name in enumerate(names):
            x, y, z = vectors_reduced[i]
            ax.scatter(x, y, z, s=100, label=name)
            ax.text(x, y, z, name, fontsize=12, ha='right', va='bottom')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('3D Visualization of Function Embeddings')
    elif n_components == 2:
        for i, name in enumerate(names):
            x, y = vectors_reduced[i]
            plt.scatter(x, y, s=100, label=name)
            plt.text(x, y, name, fontsize=12, ha='right', va='bottom')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('2D Visualization of Function Embeddings')
    else:
        for i, name in enumerate(names):
            x = vectors_reduced[i][0]
            plt.scatter(x, 0, s=100, label=name)
            plt.text(x, 0, name, fontsize=12, ha='right', va='bottom')
        plt.xlabel('PC1')
        plt.yticks([])
        plt.title('1D Visualization of Function Embeddings')
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img = Image.open(buf)

    # compute similarities
    sim_df = compute_similarities(names, vectors)
    sim_df["Cosine Similarity"] = sim_df["Cosine Similarity"].round(4)
    sim_df["Euclidean Distance"] = sim_df["Euclidean Distance"].round(4)

    return info_df, img, sim_df

description = """
# Function Embedding Visualize

Upload your `embeddings.tsv` file (function names in the first column, embedding values in the rest).
The top table shows loaded vector names and all their dimension values.
Below, you will see a plot (3D if possible) and a table with cosine similarity and Euclidean distance between each pair of vectors.
"""

with gr.Blocks(theme="default") as demo:
    gr.Markdown(description)
    file_input = gr.File(label="Upload embeddings.tsv")
    info_table = gr.Dataframe(label="Loaded Vectors (Name & Dimensions)", interactive=False)
    with gr.Row():
        plot = gr.Image(type="pil", label="Embedding Plot")
        sim_table = gr.Dataframe(label="Vector Similarity Table", interactive=False)

    def process(file):
        return plot_embeddings(file)

    file_input.change(
        process,
        inputs=file_input,
        outputs=[info_table, plot, sim_table]
    )

if __name__ == "__main__":
    demo.launch()