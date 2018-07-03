import time
from time import gmtime, strftime

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.colors as colors

import torch 
from torch.autograd import Variable
from torch import autograd, optim, nn

from sklearn.neighbors import NearestNeighbors

import numpy as np
import matplotlib.pyplot as plt
import operator

import environment as eu
import MF as mf
import EC as ec

