import tensorflow.compat.v1 as tf
print(tf.__version__)
import sys
sys.path.append("..")
import numpy as np
from datetime import datetime
from gnn_siamese import GraphSiameseNetwork
from utilities import *
import os
import argparse
import json
import time
from collections import defaultdict
import multiprocessing

tf.disable_eager_execution()

# functions for input batching

def pad_feature_matrices(matrices, pad_value=0):
    """Pad a list of feature matrices to the same shape."""
    max_nodes = max(m.shape[0] for m in matrices)
    feature_dim = matrices[0].shape[1]
    padded = np.full((len(matrices), max_nodes, feature_dim), pad_value, dtype=matrices[0].dtype)
    for i, m in enumerate(matrices):
        padded[i, :m.shape[0], :m.shape[1]] = m
    return padded

def pad_adj_matrices(matrices, pad_value=0):
    """Pad a list of adjacency matrices to the same shape."""
    max_nodes = max(m.shape[0] for m in matrices)
    padded = np.full((len(matrices), max_nodes, max_nodes), pad_value, dtype=matrices[0].dtype)
    for i, m in enumerate(matrices):
        padded[i, :m.shape[0], :m.shape[1]] = m
    return padded

def group_targets_by_size(target_json_list):
    """Group target functions by their number of nodes (basic blocks)."""
    size_groups = defaultdict(list)
    for t in target_json_list:
        num_nodes = len(t["feature_list"])
        size_groups[num_nodes].append(t)
    return size_groups

# main worker

def persistent_worker(input_queue, output_queue, size_groups, batch_size, model_args):
    """Worker process: loads model once, processes tasks from input_queue, writes results to output_queue."""
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
    import numpy as np
    from gnn_siamese import GraphSiameseNetwork

    # load model once per worker
    model = GraphSiameseNetwork(
        N_x=model_args['NODE_FEATURE_DIM'],
        Dtype=model_args['Dtype'],
        N_embed=model_args['EMBED_DIM'],
        depth_embed=model_args['EMBED_DEPTH'],
        N_o=model_args['OUTPUT_DIM'],
        ITER_LEVEL=model_args['ITERATION_LEVEL'],
        lr=model_args['LEARNING_RATE']
    )
    model.initialize_or_restore(model_args['LOAD_PATH'], model_args['LOG_PATH'])

    while True:
        lib_dic = input_queue.get()
        if lib_dic is None:
            break

        lib_func_name = lib_dic['func_name']
        lib_feat = np.asarray(lib_dic["feature_list"])
        lib_adj = np.asarray(lib_dic["adjacent_matrix"])
        lib_nodes = lib_feat.shape[0]
        lib_feat_dim = lib_feat.shape[1]

        similarities = []
        for num_nodes, group in size_groups.items():
            if lib_nodes > num_nodes:
                continue  # skip this group

            for batch_start in range(0, len(group), batch_size):
                batch_targets = group[batch_start:batch_start+batch_size]
                batch_size_actual = len(batch_targets)
                X2_list = [np.asarray(t["feature_list"]) for t in batch_targets]
                mask2_list = [np.asarray(t["adjacent_matrix"]) for t in batch_targets]
                X2 = pad_feature_matrices(X2_list)
                mask2 = pad_adj_matrices(mask2_list)

                # pad lib (second input) function to match group size and batch
                cve_feat_broadcast = np.broadcast_to(lib_feat, (batch_size_actual, lib_nodes, lib_feat_dim))
                cve_feat_padded = np.zeros((batch_size_actual, num_nodes, lib_feat_dim))
                cve_feat_padded[:, :lib_nodes, :lib_feat_dim] = cve_feat_broadcast

                cve_adj_broadcast = np.broadcast_to(lib_adj, (batch_size_actual, lib_nodes, lib_nodes))
                cve_adj_padded = np.zeros((batch_size_actual, num_nodes, num_nodes))
                cve_adj_padded[:, :lib_nodes, :lib_nodes] = cve_adj_broadcast

                sim = model.compute_similarity(
                    X1=cve_feat_padded, X2=X2, mask1=cve_adj_padded, mask2=mask2
                )
                for i, s in enumerate(sim):
                    similarities.append((-s, batch_targets[i]['func_name'])) # notice: reversed similarity value

        # top 5 most similar functions
        top5 = sorted(similarities, key=lambda x: x[0], reverse=True)[:5] # notice: reverse true
        output_queue.put((lib_func_name, [
            {"similarity": float(score), "target_func_name": name} for score, name in top5
        ]))


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0', help='visible gpu device')
parser.add_argument('--fea_dim', type=int, default=7, help='feature dimension')
parser.add_argument('--embed_dim', type=int, default=64, help='embedding dimension')
parser.add_argument('--embed_depth', type=int, default=2, help='embedding network depth')
parser.add_argument('--output_dim', type=int, default=64, help='output layer dimension')
parser.add_argument('--iter_level', type=int, default=5, help='iteration times')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for smarter batching')
parser.add_argument('--load_path', type=str, default='../saved_model/graphnn-model_best', help='path for model loading, "#LATEST#" for the latest checkpoint')
parser.add_argument('--log_path', type=str, default=None, help='path for training log')
parser.add_argument('--num_workers', type=int, default=2, help='number of persistent worker processes')
parser.add_argument("lib_function_json", type=str, help="the json file saves the feature list and adjacent matrix for one function")
parser.add_argument("target_function_json", type=str, help="the json file saves the feature lists and adjacent matrix for many function")


if __name__ == '__main__':
    args = parser.parse_args()
    args.dtype = tf.float32
    print("==============")
    print(args)
    print("==============")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    # model and data parameters
    model_args = {
        'NODE_FEATURE_DIM': args.fea_dim,
        'Dtype': args.dtype,
        'EMBED_DIM': args.embed_dim,
        'EMBED_DEPTH': args.embed_depth,
        'OUTPUT_DIM': args.output_dim,
        'ITERATION_LEVEL': args.iter_level,
        'LEARNING_RATE': args.lr,
        'LOAD_PATH': args.load_path,
        'LOG_PATH': args.log_path
    }
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers

    # load input data
    with open(args.lib_function_json, 'r') as f:
        cve_dic_list = json.load(f)
    with open(args.target_function_json, 'r') as f:
        target_json_list = json.load(f)

    # group targets by graph size for efficient batching
    size_groups = group_targets_by_size(target_json_list)
    start_time = time.time()
    results = {}

    # multiprocessing setup (todo: revert for testing)
    input_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()
    workers = []
    for _ in range(NUM_WORKERS):
        p = multiprocessing.Process(
            target=persistent_worker,
            args=(input_queue, output_queue, size_groups, BATCH_SIZE, model_args)
        )
        p.start()
        workers.append(p)

    # feed tasks to workers
    for cve_dic in cve_dic_list:
        input_queue.put(cve_dic)
    for _ in range(NUM_WORKERS):
        input_queue.put(None)

    # collect results
    for idx in range(len(cve_dic_list)):
        cve_func_name, top5 = output_queue.get()
        results[cve_func_name] = top5
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {idx + 1} functions, elapsed time: {elapsed:.2f} seconds")

    for p in workers:
        p.join()

    # save results with timestamp
    saved_file_name = f"similarity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    print(f"Saving results to {saved_file_name}")
    with open(saved_file_name, 'w') as f:
        json.dump(results, f, indent=2)