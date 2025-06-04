import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from sklearn.decomposition import PCA
import tensorflow.compat.v1 as tf
import json
from gnn_siamese import GraphSiameseNetwork

tf.disable_eager_execution()

def pad_feature_matrices(matrices, pad_value=0):
    max_nodes = max(m.shape[0] for m in matrices)
    feature_dim = matrices[0].shape[1]
    padded = np.full((len(matrices), max_nodes, feature_dim), pad_value, dtype=matrices[0].dtype)
    for i, m in enumerate(matrices):
        padded[i, :m.shape[0], :m.shape[1]] = m
    return padded

def pad_adj_matrices(matrices, pad_value=0):
    max_nodes = max(m.shape[0] for m in matrices)
    padded = np.full((len(matrices), max_nodes, max_nodes), pad_value, dtype=matrices[0].dtype)
    for i, m in enumerate(matrices):
        padded[i, :m.shape[0], :m.shape[1]] = m
    return padded

def load_graph_json(json_file):
    with open(json_file.name, "r") as f:
        data = json.load(f)
    return data

def get_embeddings(model, func_list):
    features = [np.asarray(f["feature_list"]) for f in func_list]
    adjs = [np.asarray(f["adjacent_matrix"]) for f in func_list]
    X = pad_feature_matrices(features)
    mask = pad_adj_matrices(adjs)
    embeddings = model.get_embedding(X, mask)
    return embeddings

def plot_embeddings(names, embeddings):
    n_samples, n_features = embeddings.shape
    n_components = min(3, n_samples, n_features)
    pca = PCA(n_components=n_components)
    vectors_reduced = pca.fit_transform(embeddings)

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
        ax.set_title('3D визуализация векторных представлений')
    elif n_components == 2:
        for i, name in enumerate(names):
            x, y = vectors_reduced[i]
            plt.scatter(x, y, s=100, label=name)
            plt.text(x, y, name, fontsize=12, ha='right', va='bottom')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('2D визуализация векторных представлений')
    else:
        for i, name in enumerate(names):
            x = vectors_reduced[i][0]
            plt.scatter(x, 0, s=100, label=name)
            plt.text(x, 0, name, fontsize=12, ha='right', va='bottom')
        plt.xlabel('PC1')
        plt.yticks([])
        plt.title('1D визуализация векторных представлений')
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img = Image.open(buf)
    return img

def compute_all_pair_similarities(names, embeddings, file_ids):
    results = []
    n = len(names)
    for i in range(n):
        for j in range(n):
            if file_ids[i] == file_ids[j]:
                continue
            cos_sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
            results.append({
                "Function 1": names[i],
                "File 1": os.path.basename(file_ids[i]),
                "Function 2": names[j],
                "File 2": os.path.basename(file_ids[j]),
                "Cosine Similarity": cos_sim
            })
    return pd.DataFrame(results)

description = """
# Визуализация сходства функций на основе модели (inference)

Входные данные: любое количество файлов JSON (файлов с ACFG функций).
Алгоритм вычислит вектора для всех функций и отобразит:
- Таблицу имен функций, их исходный файл и измерения.
- Изображение векторных представлений (PCA).
- Таблицу результатов предсказания схожести между всеми парами (по всем файлам).
"""

# Model parameters (edit as needed)
MODEL_PARAMS = {
    'NODE_FEATURE_DIM': 7,
    'Dtype': tf.float32,
    'EMBED_DIM': 64,
    'EMBED_DEPTH': 2,
    'OUTPUT_DIM': 64,
    'ITERATION_LEVEL': 5,
    'LEARNING_RATE': 1e-4,
    'LOAD_PATH': './saved_model/graphnn-model_best',
    'LOG_PATH': None
}

with gr.Blocks(theme="default") as demo:
    gr.Markdown(description)
    file_inputs = gr.Files(label="Загрузить JSON файлы")
    info_table = gr.Dataframe(label="Raw-вектора функций", interactive=False)
    plot = gr.Image(type="pil", label="Плоттинг векторов")
    sim_table = gr.Dataframe(label="Результат similarity inference", interactive=False)

    def process(files):
        if not files or len(files) == 0:
            return pd.DataFrame(), None, pd.DataFrame()

        # Load model once per inference
        model = GraphSiameseNetwork(
            N_x=MODEL_PARAMS['NODE_FEATURE_DIM'],
            Dtype=MODEL_PARAMS['Dtype'],
            N_embed=MODEL_PARAMS['EMBED_DIM'],
            depth_embed=MODEL_PARAMS['EMBED_DEPTH'],
            N_o=MODEL_PARAMS['OUTPUT_DIM'],
            ITER_LEVEL=MODEL_PARAMS['ITERATION_LEVEL'],
            lr=MODEL_PARAMS['LEARNING_RATE']
        )
        model.initialize_or_restore(MODEL_PARAMS['LOAD_PATH'], MODEL_PARAMS['LOG_PATH'])

        all_names = []
        all_embs = []
        all_file_ids = []
        info_data = []

        for file in files:
            func_list = load_graph_json(file)
            names = [f["func_name"] for f in func_list]
            embs = get_embeddings(model, func_list)
            file_short = os.path.basename(file.name)
            all_names.extend(names)
            all_embs.append(embs)
            all_file_ids.extend([file_short] * len(names))
            for name, vec in zip(names, embs):
                info_data.append([file_short, name] + list(vec))

        all_embs = np.vstack(all_embs)
        info_df = pd.DataFrame(info_data, columns=["File", "Name"] + [f"Dim {i+1}" for i in range(all_embs.shape[1])])
        img = plot_embeddings(all_names, all_embs)
        sim_df = compute_all_pair_similarities(all_names, all_embs, all_file_ids)
        sim_df["Cosine Similarity"] = sim_df["Cosine Similarity"].round(4)
        return info_df, img, sim_df

    file_inputs.change(
        process,
        inputs=file_inputs,
        outputs=[info_table, plot, sim_table]
    )

if __name__ == "__main__":
    demo.launch()