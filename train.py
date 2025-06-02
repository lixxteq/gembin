import tensorflow.compat.v1 as tf
print(tf.__version__)
import numpy as np
from datetime import datetime
from gnn_siamese import GraphSiameseNetwork
from utilities import *
import os
import argparse
import json

tf.disable_eager_execution()

# model argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='-1', help='visible gpu device')
parser.add_argument('--fea_dim', type=int, default=7, help='feature dimension')
parser.add_argument('--embed_dim', type=int, default=64, help='embedding dimension')
parser.add_argument('--embed_depth', type=int, default=2, help='embedding network depth')
parser.add_argument('--output_dim', type=int, default=64, help='output layer dimension')
parser.add_argument('--iter_level', type=int, default=5, help='iteration times')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--batch_size', type=int, default=5, help='batch size')
parser.add_argument('--load_path', type=str, default=None, help='path for model loading, "#LATEST#" for the latest checkpoint')
parser.add_argument('--save_path', type=str, default='./saved_model/graphnn-model', help='path for model saving')
parser.add_argument('--log_path', type=str, default=None, help='path for training log')

if __name__ == '__main__':
    args = parser.parse_args()
    args.dtype = tf.float32
    print(f"Model arguments: {args}")

    # env and hyperparms. SHOULD BE EDITED for custom datasets
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    Dtype = args.dtype
    NODE_FEATURE_DIM = args.fea_dim
    EMBED_DIM = args.embed_dim
    EMBED_DEPTH = args.embed_depth
    OUTPUT_DIM = args.output_dim
    ITERATION_LEVEL = args.iter_level
    LEARNING_RATE = args.lr
    MAX_EPOCH = args.epoch
    BATCH_SIZE = args.batch_size
    LOAD_PATH = args.load_path
    SAVE_PATH = args.save_path
    LOG_PATH = args.log_path

    SHOW_FREQ = 1
    TEST_FREQ = 1
    SAVE_FREQ = 5
    DATA_FILE_NAME = f'./data/acfgSSL_{NODE_FEATURE_DIM}/'
    SOFTWARE = ('openssl-1.0.1f-', 'openssl-1.0.1u-')
    OPTIMIZATION = ('-O0', '-O1', '-O2', '-O3')
    COMPILER = ('armeb-linux', 'i586-linux', 'mips-linux')
    VERSION = ('v54',)

    # prepare dataset
    # generate filenames and function name dictionary
    filenames = generate_filenames(DATA_FILE_NAME, SOFTWARE, COMPILER, OPTIMIZATION, VERSION)
    funcname_dict = generate_funcname_dict(filenames)

    # read graphs and class assignments
    graphs, classes = read_graphs(filenames, funcname_dict, NODE_FEATURE_DIM)
    print(f"{len(graphs)} graphs, {len(classes)} functions")

    # partition data into train/dev/test (todo: not random permutation?)
    if os.path.isfile('data/class_perm.npy'):
        perm = np.load('data/class_perm.npy')
    else:
        perm = np.random.permutation(len(classes))
        np.save('data/class_perm.npy', perm)
    if len(perm) < len(classes):
        perm = np.random.permutation(len(classes))
        np.save('data/class_perm.npy', perm)

    # partition graphs and classes
    Gs_train, classes_train, Gs_dev, classes_dev, Gs_test, classes_test = partition_graphs(
        graphs, classes, [0.8, 0.1, 0.1], perm)

    print(f"Train: {len(Gs_train)} graphs, {len(classes_train)} functions")
    print(f"Dev: {len(Gs_dev)} graphs, {len(classes_dev)} functions")
    print(f"Test: {len(Gs_test)} graphs, {len(classes_test)} functions")

    # prepare Validation Pairs
    if os.path.isfile('data/valid.json'):
        with open('data/valid.json') as inf:
            valid_ids = json.load(inf)
        valid_epoch = generate_epoch_pairs(Gs_dev, classes_dev, BATCH_SIZE, load_id=valid_ids)
    else:
        valid_epoch, valid_ids = generate_epoch_pairs(Gs_dev, classes_dev, BATCH_SIZE, output_id=True)
        with open('data/valid.json', 'w') as outf:
            json.dump(valid_ids, outf)

    # model initialization
    gnn = GraphSiameseNetwork(
        N_x=NODE_FEATURE_DIM,
        Dtype=Dtype,
        N_embed=EMBED_DIM,
        depth_embed=EMBED_DEPTH,
        N_o=OUTPUT_DIM,
        ITER_LEVEL=ITERATION_LEVEL,
        lr=LEARNING_RATE
    )
    gnn.initialize_or_restore(LOAD_PATH, LOG_PATH)

    # initial AUC eval
    auc, fpr, tpr, thres = get_auc_epoch(gnn, Gs_train, classes_train, BATCH_SIZE, load_data=valid_epoch)
    gnn.log_message(f"Initial training auc = {auc} @ {datetime.now()}")
    auc0, fpr, tpr, thres = get_auc_epoch(gnn, Gs_dev, classes_dev, BATCH_SIZE, load_data=valid_epoch)
    gnn.log_message(f"Initial validation auc = {auc0} @ {datetime.now()}")

    # training loop
    best_auc = 0
    for i in range(1, MAX_EPOCH + 1):
        loss = train_epoch(gnn, Gs_train, classes_train, BATCH_SIZE)
        gnn.log_message(f"EPOCH {i}/{MAX_EPOCH}, loss = {loss} @ {datetime.now()}")

        if i % TEST_FREQ == 0:
            auc, _, _, _ = get_auc_epoch(gnn, Gs_train, classes_train, BATCH_SIZE, load_data=valid_epoch)
            gnn.log_message(f"Testing model: training auc = {auc} @ {datetime.now()}")
            auc, _, _, _ = get_auc_epoch(gnn, Gs_dev, classes_dev, BATCH_SIZE, load_data=valid_epoch)
            gnn.log_message(f"Testing model: validation auc = {auc} @ {datetime.now()}")

            if auc > best_auc:
                path = gnn.save_checkpoint(SAVE_PATH + '_best')
                best_auc = auc
                gnn.log_message(f"Model saved in {path}")

        if i % SAVE_FREQ == 0:
            path = gnn.save_checkpoint(SAVE_PATH, i)
            gnn.log_message(f"Model saved in {path}")
