import numpy as np
from sklearn.metrics import auc, roc_curve
import json

def generate_filenames(DATA, SF, CM, OP, VS):
    """Generate all possible filenames for the dataset. ONLY for openssl dataset, should be chaged for custom datasets"""
    return [f"{DATA}{sf}{cm}{op}{vs}.json" for sf in SF for cm in CM for op in OP for vs in VS]

def generate_funcname_dict(filenames):
    """Create a mapping from function name to unique index."""
    name_dict = {}
    name_num = 0
    for fname in filenames:
        with open(fname) as inf:
            for line in inf:
                g_info = json.loads(line.strip())
                if g_info['fname'] not in name_dict:
                    name_dict[g_info['fname']] = name_num
                    name_num += 1
    return name_dict

class Graph(object):
    """Graph structure for storing nodes, features, and edges."""
    def __init__(self, node_num=0, label=None, name=None):
        self.node_num = node_num
        self.label = label
        self.name = name
        self.features = [[] for _ in range(node_num)]
        self.succs = [[] for _ in range(node_num)]
        self.preds = [[] for _ in range(node_num)]

    def add_node(self, feature=None):
        """Add a node with features."""
        if feature is None:
            feature = []
        self.node_num += 1
        self.features.append(feature)
        self.succs.append([])
        self.preds.append([])

    def add_edge(self, u, v):
        """Add a directed edge from u to v."""
        self.succs[u].append(v)
        self.preds[v].append(u)

    def to_string(self):
        """Return a string representation of the graph."""
        ret = f'{self.node_num} {self.label}\n'
        for u in range(self.node_num):
            ret += ' '.join(str(fea) for fea in self.features[u]) + ' '
            ret += str(len(self.succs[u]))
            for succ in self.succs[u]:
                ret += f' {succ}'
            ret += '\n'
        return ret

def read_graphs(filenames, funcname_dict, feature_dim):
    """
    Read graphs from files and assign them to classes.
    Returns a list of Graph objects and a list of class indices.
    """
    graphs = []
    classes = [[] for _ in range(len(funcname_dict))] if funcname_dict is not None else []
    for fname in filenames:
        with open(fname) as inf:
            for line in inf:
                g_info = json.loads(line.strip())
                label = funcname_dict[g_info['fname']]
                classes[label].append(len(graphs))
                cur_graph = Graph(g_info['n_num'], label, g_info['src'])
                for u in range(g_info['n_num']):
                    cur_graph.features[u] = np.array(g_info['features'][u])
                    for v in g_info['succs'][u]:
                        cur_graph.add_edge(u, v)
                graphs.append(cur_graph)
    return graphs, classes

def partition_graphs(graphs, classes, partitions, perm):
    """
    Partition graphs and classes according to the given partition sizes and permutation.
    """
    C = len(classes)
    st = 0.0
    ret = []
    for part in partitions:
        cur_g = []
        cur_c = []
        ed = st + part * C
        for cls in range(int(st), int(ed)):
            prev_class = classes[perm[cls]]
            cur_c.append([])
            for i in range(len(prev_class)):
                cur_g.append(graphs[prev_class[i]])
                cur_g[-1].label = len(cur_c) - 1
                cur_c[-1].append(len(cur_g) - 1)
        ret.append(cur_g)
        ret.append(cur_c)
        st = ed
    return ret

def generate_epoch_pairs(graphs, classes, batch_size, output_id=False, load_id=None):
    """
    Generate positive and negative graph pairs for an epoch.
    """
    epoch_data = []
    id_data = []
    if load_id is None:
        st = 0
        while st < len(graphs):
            if output_id:
                X1, X2, m1, m2, y, pos_id, neg_id = get_graph_pairs(graphs, classes, batch_size, st=st, output_id=True)
                id_data.append((pos_id, neg_id))
            else:
                X1, X2, m1, m2, y = get_graph_pairs(graphs, classes, batch_size, st=st)
            epoch_data.append((X1, X2, m1, m2, y))
            st += batch_size
    else:
        id_data = load_id
        for id_pair in id_data:
            X1, X2, m1, m2, y = get_graph_pairs(graphs, classes, batch_size, load_id=id_pair)
            epoch_data.append((X1, X2, m1, m2, y))
    return (epoch_data, id_data) if output_id else epoch_data

def get_graph_pairs(graphs, classes, batch_size, st=-1, output_id=False, load_id=None):
    """
    Generate positive and negative graph pairs for training or evaluation.
    """
    if load_id is None:
        C = len(classes)
        if st + batch_size > len(graphs):
            batch_size = len(graphs) - st
        ed = st + batch_size
        pos_ids = []
        neg_ids = []
        for g_id in range(st, ed):
            g0 = graphs[g_id]
            cls = g0.label
            tot_g = len(classes[cls])
            if len(classes[cls]) >= 2:
                g1_id = classes[cls][np.random.randint(tot_g)]
                while g_id == g1_id:
                    g1_id = classes[cls][np.random.randint(tot_g)]
                pos_ids.append((g_id, g1_id))
            cls2 = np.random.randint(C)
            while len(classes[cls2]) == 0 or cls2 == cls:
                cls2 = np.random.randint(C)
            tot_g2 = len(classes[cls2])
            h_id = classes[cls2][np.random.randint(tot_g2)]
            neg_ids.append((g_id, h_id))
    else:
        pos_ids, neg_ids = load_id

    M_pos = len(pos_ids)
    M_neg = len(neg_ids)
    M = M_pos + M_neg

    maxN1 = max([graphs[pair[0]].node_num for pair in pos_ids + neg_ids])
    maxN2 = max([graphs[pair[1]].node_num for pair in pos_ids + neg_ids])
    feature_dim = len(graphs[0].features[0])

    X1_input = np.zeros((M, maxN1, feature_dim))
    X2_input = np.zeros((M, maxN2, feature_dim))
    node1_mask = np.zeros((M, maxN1, maxN1))
    node2_mask = np.zeros((M, maxN2, maxN2))
    y_input = np.zeros((M))

    for i in range(M_pos):
        y_input[i] = 1
        g1 = graphs[pos_ids[i][0]]
        g2 = graphs[pos_ids[i][1]]
        for u in range(g1.node_num):
            X1_input[i, u, :] = np.array(g1.features[u])
            for v in g1.succs[u]:
                node1_mask[i, u, v] = 1
        for u in range(g2.node_num):
            X2_input[i, u, :] = np.array(g2.features[u])
            for v in g2.succs[u]:
                node2_mask[i, u, v] = 1

    for i in range(M_pos, M_pos + M_neg):
        y_input[i] = -1
        g1 = graphs[neg_ids[i - M_pos][0]]
        g2 = graphs[neg_ids[i - M_pos][1]]
        for u in range(g1.node_num):
            X1_input[i, u, :] = np.array(g1.features[u])
            for v in g1.succs[u]:
                node1_mask[i, u, v] = 1
        for u in range(g2.node_num):
            X2_input[i, u, :] = np.array(g2.features[u])
            for v in g2.succs[u]:
                node2_mask[i, u, v] = 1

    if output_id:
        return X1_input, X2_input, node1_mask, node2_mask, y_input, pos_ids, neg_ids
    else:
        return X1_input, X2_input, node1_mask, node2_mask, y_input

def train_epoch(model, graphs, classes, batch_size, load_data=None):
    """
    Train the model for one epoch and return average loss.
    """
    if load_data is None:
        epoch_data = generate_epoch_pairs(graphs, classes, batch_size)
    else:
        epoch_data = load_data
    perm = np.random.permutation(len(epoch_data))
    cum_loss = 0.0
    for index in perm:
        X1, X2, mask1, mask2, y = epoch_data[index]
        loss = model.train_step(X1, X2, mask1, mask2, y)
        cum_loss += loss
    return cum_loss / len(perm)

def get_auc_epoch(model, graphs, classes, batch_size, load_data=None):
    """
    Evaluate model AUC for one epoch.
    """
    tot_diff = []
    tot_truth = []
    if load_data is None:
        epoch_data = generate_epoch_pairs(graphs, classes, batch_size)
    else:
        epoch_data = load_data
    for X1, X2, m1, m2, y in epoch_data:
        diff = model.compute_similarity(X1, X2, m1, m2)
        tot_diff += list(diff)
        tot_truth += list(y > 0)
    diff = np.array(tot_diff)
    truth = np.array(tot_truth)
    fpr, tpr, thres = roc_curve(truth, (1 - diff) / 2)
    model_auc = auc(fpr, tpr)
    return model_auc, fpr, tpr, thres