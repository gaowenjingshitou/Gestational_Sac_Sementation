#from Load_Inhosp_No_divided_v1 import Data
from efficientnet_pytorch import EfficientNet
import pretrainedmodels
from torch import optim, nn
#from Load_data_10_14 import Data
from Load_standard_plane_0216_Test import Data

#from Load_data_10_19_NEC_2007 import Data
#from Load_data_10_19_surgery import Data
from pandas.core.frame import DataFrame
from resnest.torch import resnest50
# from    resnet import ResNet18
from torchvision.models import resnet18
from torchvision.models import resnet50
import torch

import random
from torch import optim, nn
import seaborn as sns
import visdom
import torchvision
from sklearn import metrics
from resnest.torch import resnest50
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from torchvision import models

import numpy as np
from sklearn import metrics

import time
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from efficientnet_pytorch import EfficientNet
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix
from util import Flatten

#seed=3 #诊断的seed
#手术的
seed=3
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
def _init_fn(worker_id):
    np.random.seed(int(seed))


device = torch.device('cuda')
print("device=",device)

batchsz = 32
epochs =50

print("*******Start training******")
print("共{}epoch".format(epochs))


