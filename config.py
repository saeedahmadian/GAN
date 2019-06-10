import numpy as np
import tensorflow as tf


mydata=np.load('traindata.npy')
nsampel=len(mydata)
nnodes=1
nfeature=len(mydata[0,0,:,0])


flags = tf.app.flags
flags.DEFINE_string("mode", "train", "train or test")

# data paramters
flags.DEFINE_integer("num_samples", nsampel, "number of samples")
flags.DEFINE_integer("num_nodes", nnodes, "number of nodes")
flags.DEFINE_integer("num_features", nfeature, "number of features")
#flags.DEFINE_string("data_dir", "data", "directory name to read the data")



# training parameters
flags.DEFINE_integer("epoch", 1000, "number of epochs t train")
flags.DEFINE_float("learning_rate", 0.02, "learning rate of optimizer")
flags.DEFINE_float("momentum", 0.2, "momentum  of optimizer")
flags.DEFINE_integer("batch_size", 64, "batch size for training")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "directory name to save the checkpoints")

args = flags.FLAGS
