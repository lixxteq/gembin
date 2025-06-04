import tensorflow.compat.v1 as tf
import numpy as np
import argparse
import json
import os
from gnn_siamese import GraphSiameseNetwork
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

tf.disable_eager_execution()

def pad_feature_matrix(matrix, target_nodes, pad_value=0):
    """pad a feature matrix to target_nodes rows."""
    feature_dim = matrix.shape[1]
    padded = np.full((1, target_nodes, feature_dim), pad_value, dtype=matrix.dtype)
    padded[0, :matrix.shape[0], :matrix.shape[1]] = matrix
    return padded

def pad_adj_matrix(matrix, target_nodes, pad_value=0):
    """pad an adjacency matrix to target_nodes x target_nodes."""
    padded = np.full((1, target_nodes, target_nodes), pad_value, dtype=matrix.dtype)
    padded[0, :matrix.shape[0], :matrix.shape[1]] = matrix
    return padded

def load_function_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # If the file contains a list, take the first element
    if isinstance(data, list):
        data = data[0]
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("func_json", nargs='+', type=str, help="list of JSON files with functions ACFG")
    parser.add_argument('--fea_dim', type=int, default=7, help='feature dimension')
    parser.add_argument('--embed_dim', type=int, default=64, help='embedding dimension')
    parser.add_argument('--embed_depth', type=int, default=2, help='embedding network depth')
    parser.add_argument('--output_dim', type=int, default=64, help='output layer dimension')
    parser.add_argument('--iter_level', type=int, default=5, help='iteration times')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--load_path', type=str, default='./saved_model/graphnn-model_best', help='path for model loading')
    parser.add_argument('--log_path', type=str, default=None, help='path for training log')
    parser.add_argument('--output_tsv', type=str, default='embeddings.tsv', help='Output TSV file')
    parser.add_argument('--device', type=str, default='0', help='visible gpu device')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    args.dtype = tf.float32

    nc = []
    func = []
    # load function graphs
    for f in args.func_json:
        func = load_function_json(f)

        # determine max node count for padding
        n1 = len(func["feature_list"])
        nc.append(n1)

    max_nodes = max(nc)

    # load model
    model = GraphSiameseNetwork(
        N_x=args.fea_dim,
        Dtype=args.dtype,
        N_embed=args.embed_dim,
        depth_embed=args.embed_depth,
        N_o=args.output_dim,
        ITER_LEVEL=args.iter_level,
        lr=args.lr
    )
    model.initialize_or_restore(args.load_path, args.log_path)
    
    # write to TSV
    with open(args.output_tsv, 'w') as f:
        f.write("function\t" + "\t".join([f"dim{i+1}" for i in range(args.embed_dim)]) + "\n")

    with open(args.output_tsv, 'a') as file:
        for f in func:
            # prepare feature and adjacency matrices
            X1 = pad_feature_matrix(np.asarray(f["feature_list"]), max_nodes)
            mask1 = pad_adj_matrix(np.asarray(f["adjacent_matrix"]), max_nodes)

            # get embedding
            emb1 = model.get_embedding(X1, mask1)[0]

            file.write(f"{f.get('func_name', 'func.any')}\t" + "\t".join(map(str, emb1)) + "\n")

    print(f"Embeddings saved to {args.output_tsv}")

if __name__ == "__main__":
    main()