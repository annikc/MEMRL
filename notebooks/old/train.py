import numpy as np
import sys
import argparse

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('savedir',type=str, help='Prefix of folder name where data will be saved')
parser.add_argument("-n_epochs", type=int, help="Number of epochs", default=5000)
parser.add_argument("-n_events", type=int, help="Number of events per epoch", default=100)
parser.add_argument("eta", type=float, help="Network learning rate", default=)


parser.add_argument("-store", type=int, help="Frequency (ie. every ___ examples) at which to save network state & get test error", default=1000)
parser.add_argument("-n_layers", type=int, help="Number of layers", default=4)
parser.add_argument("-n_units", help="Number of units in each layer", type=lambda s: [int(item) for item in s.split(',')], default=[784, 500, 500, 10])
parser.add_argument("-W_range", help="Range of initial feedforward weights", type=lambda s: [float(item) for item in s.split(',')], default=[0, 0.1, 0.1, 0.01])
parser.add_argument("-Z_range", help="Range of initial recurrent weights", type=lambda s: [float(item) for item in s.split(',')], default=[0, 0.01, 0.01])
parser.add_argument("-Y_range", help="Range of initial feedback weights", type=lambda s: [float(item) for item in s.split(',')], default=[0, 1.0, 1.0])
parser.add_argument("-f_etas", help="Feedforward learning rates", type=lambda s: [float(item) for item in s.split(',')], default=[0, 0.5, 0.5, 0.01])
parser.add_argument("-r_etas", help="Recurrent learning rates", type=lambda s: [float(item) for item in s.split(',')], default=[0, 0, 0])
parser.add_argument("-b_etas", help="Feedback learning rates", type=lambda s: [float(item) for item in s.split(',')], default=[0, 0, 0])
parser.add_argument("-output_burst_prob", type=float, help="Output layer burst probability", default=0.2)
parser.add_argument("-desired_u", type=float, help="Desired apical potential", default=0.05)
parser.add_argument("-use_tensorboard", type=bool, help="Whether to use Tensorboard", default=False)
parser.add_argument("-hard_m", type=float, help="Hard derivative mean", default=0.1)
parser.add_argument("-hard_v", type=float, help="Hard derivative variance", default=0.05)
parser.add_argument("-info", type=str, help="Any other information about the experiment", default="")
parser.add_argument("-symmetric_weights", type=bool, help="Whether to use symmetric weights", default=False)
parser.add_argument("-momentum", type=float, help="Momentum", default=0)
args=parser.parse_args()

# set network parameters
import network as net
net.n_epochs          = args.n_epochs
net.store             = args.store
net.n_layers          = args.n_layers
net.n_units           = args.n_units
net.W_range           = args.W_range
net.Z_range           = args.Z_range
net.Y_range           = args.Y_range
net.f_etas            = args.f_etas
net.r_etas            = args.r_etas
net.b_etas            = args.b_etas
net.output_burst_prob = args.output_burst_prob
net.desired_u         = args.desired_u
net.use_tensorboard   = args.use_tensorboard
net.hard_m            = args.hard_m 
net.hard_v            = args.hard_v 
net.info              = args.info
net.symmetric_weights = args.symmetric_weights
net.momentum          = args.momentum

# train the network
net.train(folder_prefix=args.folder_prefix)