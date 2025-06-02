import tensorflow.compat.v1 as tf
import numpy as np
import datetime
from sklearn.metrics import roc_auc_score

def graph_embedding(X, msg_mask, N_x, N_embed, N_o, iter_level, Wnode, Wembed, W_output, b_output):
    """
    Compute graph embeddings using message passing and neural network layers.
    """
    # Initial node embedding: affine transform of node features
    node_val = tf.reshape(tf.matmul(tf.reshape(X, [-1, N_x]), Wnode),
                          [tf.shape(X)[0], -1, N_embed])
    cur_msg = tf.nn.relu(node_val)   # [batch, node_num, embed_dim]
    for t in range(iter_level):
        # Message passing: aggregate neighbor messages
        Li_t = tf.matmul(msg_mask, cur_msg)  # [batch, node_num, embed_dim]
        # Apply neural network layers to aggregated messages
        cur_info = tf.reshape(Li_t, [-1, N_embed])
        for Wi in Wembed:
            if Wi == Wembed[-1]:
                cur_info = tf.matmul(cur_info, Wi)
            else:
                cur_info = tf.nn.relu(tf.matmul(cur_info, Wi))
        neigh_val_t = tf.reshape(cur_info, tf.shape(Li_t))
        # Add neighbor info to node embedding
        tot_val_t = node_val + neigh_val_t
        # Nonlinearity
        cur_msg = tf.nn.tanh(tot_val_t)
    # Pool node embeddings to get graph embedding
    g_embed = tf.reduce_sum(cur_msg, 1)   # [batch, embed_dim]
    # Output layer
    output = tf.matmul(g_embed, W_output) + b_output
    return output

class GraphSiameseNetwork(object):
    """
    Siamese Graph Neural Network for graph similarity.
    - Computes embeddings for two input graphs.
    - Uses cosine similarity to compare embeddings.
    - Trains with a loss that encourages similar graphs to have similar embeddings.
    """
    def __init__(self, N_x, Dtype, N_embed, depth_embed, N_o, ITER_LEVEL, lr, device='/cpu'):
        self.NODE_LABEL_DIM = N_x
        tf.reset_default_graph()
        with tf.device(device):
            # Define trainable weights for the network
            Wnode = tf.Variable(tf.truncated_normal([N_x, N_embed], stddev=0.1, dtype=Dtype))
            Wembed = [tf.Variable(tf.truncated_normal([N_embed, N_embed], stddev=0.1, dtype=Dtype))
                      for _ in range(depth_embed)]
            W_output = tf.Variable(tf.truncated_normal([N_embed, N_o], stddev=0.1, dtype=Dtype))
            b_output = tf.Variable(tf.constant(0, shape=[N_o], dtype=Dtype))
            
            # Placeholders for first graph input
            self.X1 = tf.placeholder(Dtype, [None, None, N_x])  # [B, N_node, N_x]
            self.msg1_mask = tf.placeholder(Dtype, [None, None, None])  # [B, N_node, N_node]
            # Compute embedding for first graph
            self.embed1 = graph_embedding(self.X1, self.msg1_mask, N_x, N_embed, N_o, ITER_LEVEL,
                                          Wnode, Wembed, W_output, b_output)
            # Placeholders for second graph input
            self.X2 = tf.placeholder(Dtype, [None, None, N_x])
            self.msg2_mask = tf.placeholder(Dtype, [None, None, None])
            self.embed2 = graph_embedding(self.X2, self.msg2_mask, N_x, N_embed, N_o, ITER_LEVEL,
                                          Wnode, Wembed, W_output, b_output)
            # Placeholder for label: 1 for similar, -1 for different
            self.label = tf.placeholder(Dtype, [None, ])  # same: 1; different:-1

            # Cosine similarity between the two graph embeddings
            cos = tf.reduce_sum(self.embed1 * self.embed2, 1) / (
                tf.sqrt(tf.reduce_sum(self.embed1 ** 2, 1) * tf.reduce_sum(self.embed2 ** 2, 1) + 1e-10)
            )
            # Negative cosine similarity (for loss)
            self.diff = -cos
            # Loss: mean squared error between (diff + label) and 0
            self.loss = tf.reduce_mean((self.diff + self.label) ** 2)
            # Optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

    def log_message(self, string):
        """Print and optionally log a message."""
        print(string)
        if self.log_file is not None:
            self.log_file.write(string + '\n')

    def initialize_or_restore(self, LOAD_PATH, LOG_PATH):
        """Initialize or restore model and logging."""
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        saver = tf.train.Saver()
        self.sess = sess
        self.saver = saver
        self.log_file = None
        if LOAD_PATH is not None:
            if LOAD_PATH == '#LATEST#':
                checkpoint_path = tf.train.latest_checkpoint('./')
            else:
                checkpoint_path = LOAD_PATH
            saver.restore(sess, checkpoint_path)
            if LOG_PATH is not None:
                self.log_file = open(LOG_PATH, 'a+')
            self.log_message('{}, model loaded from file: {}'.format(
                datetime.datetime.now(), checkpoint_path))
        else:
            sess.run(tf.global_variables_initializer())
            if LOG_PATH is not None:
                self.log_file = open(LOG_PATH, 'w')
            self.log_message('Training start @ {}'.format(datetime.datetime.now()))

    def get_embedding(self, X1, mask1):
        """Get embedding for a single graph."""
        vec, = self.sess.run(fetches=[self.embed1],
                             feed_dict={self.X1: X1, self.msg1_mask: mask1})
        return vec

    def compute_loss(self, X1, X2, mask1, mask2, y):
        """Calculate loss for a batch of graph pairs and labels."""
        cur_loss, = self.sess.run(fetches=[self.loss], feed_dict={
            self.X1: X1, self.X2: X2, self.msg1_mask: mask1, self.msg2_mask: mask2, self.label: y
        })
        return cur_loss

    def compute_similarity(self, X1, X2, mask1, mask2):
        """Calculate cosine similarity (diff) for a batch of graph pairs."""
        diff, = self.sess.run(fetches=[self.diff], feed_dict={
            self.X1: X1, self.X2: X2, self.msg1_mask: mask1, self.msg2_mask: mask2
        })
        return diff

    def train_step(self, X1, X2, mask1, mask2, y):
        """Perform one training step and return loss."""
        loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict={
            self.X1: X1, self.X2: X2, self.msg1_mask: mask1, self.msg2_mask: mask2, self.label: y
        })
        return loss

    def save_checkpoint(self, path, epoch=None):
        """Save model checkpoint."""
        checkpoint_path = self.saver.save(self.sess, path, global_step=epoch)
        return checkpoint_path
