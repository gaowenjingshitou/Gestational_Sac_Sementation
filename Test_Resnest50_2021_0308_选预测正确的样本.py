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


train_db = Data('Plane', 224, mode='train')
train_loader = DataLoader(train_db, batch_size=batchsz,  shuffle=True,
                          num_workers=0)

val_db = Data('Plane', 224, mode='val')
# val_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, 327)
val_loader = DataLoader(val_db, batch_size=batchsz,  shuffle=True,
                         num_workers=0)

test_db = Data('Plane', 224, mode='test')
test_loader = DataLoader(test_db, batch_size=batchsz,shuffle=True,  num_workers=0)


def Train_Test(model, loader, optimal_threshold):
    model.train()

    correct = 0
    total = len(loader.dataset)
    Inhosp_Nos, EXAM_NOs, y_trues, y_probs, y_preds = [], [], [], [], []
    for Inhosp_No, EXAM_NO, x, y in loader:
        # print("EXAM_NO",EXAM_NO)
        x, y = x.to(device), y.to(device)
        y_trues.extend(list(y.cpu().numpy()))
        Inhosp_Nos.extend(Inhosp_No)
        EXAM_NOs.extend(list(EXAM_NO))
        with torch.no_grad():
            logits = model(x)
            y_prob = logits[:, 1]

            y_prob = y_prob.detach().cpu().numpy()

            y_probs.extend(list(y_prob))

    for j in y_probs:

        if j > optimal_threshold:
            y_preds.append(1)

        else:
            y_preds.append(0)

    confusionmatrix = confusion_matrix(y_trues, y_preds)
    print('confusionmatrix', confusionmatrix)
    # pred = logits.argmax(dim=1)

    #             y_preds.extend(list(pred.detach().cpu().numpy()))

    return Inhosp_Nos, EXAM_NOs, y_trues, y_probs, y_preds, confusionmatrix


def Test(model, loader, optimal_threshold):
    model.eval()

    correct = 0
    total = len(loader.dataset)
    Inhosp_Nos, EXAM_NOs, y_trues, y_probs, preds_val = [], [], [], [], []
    for Inhosp_No, EXAM_NO, x, y in loader:
        x, y = x.to(device), y.to(device)
        y_trues.extend(list(y.cpu().numpy()))
        Inhosp_Nos.extend(Inhosp_No)
        # print("EXAM_NO",list(EXAM_NO))
        EXAM_NOs.extend(list(EXAM_NO))

        with torch.no_grad():
            logits = model(x)
            y_prob = logits[:, 1]
            y_prob = y_prob.detach().cpu().numpy()

            y_probs.extend(list(y_prob))

    for j in y_probs:

        if j > optimal_threshold:
            preds_val.append(1)

        else:
            preds_val.append(0)

    confusionmatrix_val = confusion_matrix(y_trues, preds_val)
    print('confusionmatrix_val', confusionmatrix_val)
    # pred = logits.argmax(dim=1)

    #             y_preds.extend(list(pred.detach().cpu().numpy()))

    return Inhosp_Nos, EXAM_NOs, y_trues, y_probs, preds_val, confusionmatrix_val


from keras.utils import to_categorical

from sklearn.metrics import roc_auc_score


def Train_confusion_matrixes(model, loader):
    model.train()

    correct = 0
    total = len(loader.dataset)
    y_true = []
    y_scores = []
    for Inhosp_No, EXAM_NO, x, y in loader:
        y_true.extend(y)
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_score = model(x)
            # print("y_score",y_score)

            #             pred = logits.argmax(dim=1)
            y_scores.append(y_score)

    y_true = torch.tensor([i.item() for i in y_true])

    y_true = to_categorical(y_true, 2)
    # print("y_true",y_true)
    y_scores = torch.cat([i for i in y_scores], 0)
    # print("y_scores",y_scores)

    # _scores= torch.tensor([i.item() for i in y_scores])
    #     print("y_pred",y_pred)
    #     print("y_true",y_true)
    y_scores = y_scores.cpu().numpy()
    # y_true=y_true.cpu().numpy()
    auc = roc_auc_score(y_true, y_scores)

    return auc


from keras.utils import to_categorical

from sklearn.metrics import roc_auc_score


def confusion_matrixes(model, loader):
    model.eval()

    correct = 0
    total = len(loader.dataset)
    y_true = []
    y_scores = []
    for Inhosp_No, EXAM_NO, x, y in loader:
        y_true.extend(y)
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_score = model(x)
            # print("y_score",y_score)

            #             pred = logits.argmax(dim=1)
            y_scores.append(y_score)

    y_true = torch.tensor([i.item() for i in y_true])

    y_true = to_categorical(y_true, 2)
    # print("y_true",y_true)
    y_scores = torch.cat([i for i in y_scores], 0)
    # print("y_scores",y_scores)

    # _scores= torch.tensor([i.item() for i in y_scores])
    #     print("y_pred",y_pred)
    #     print("y_true",y_true)
    y_scores = y_scores.cpu().numpy()
    # y_true=y_true.cpu().numpy()
    auc = roc_auc_score(y_true, y_scores)

    return auc


def main():
    trained_model = resnest50(pretrained=True)
    # #     trained_model = resnet18(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1],  # [b, 512, 1, 1]
                          Flatten(),  # [b, 512, 1, 1] => [b, 512]
                          # nn.Linear(2048, 512),
                          # nn.Linear(512, 2),
                          nn.BatchNorm1d(2048, affine=False),
                          nn.Dropout(0.29),
                          nn.Linear(2048, 2),
                          # nn.BatchNorm1d(2048, affine=False),
                          # nn.Linear(248, 2),
                          nn.Softmax(dim=1)
                          ).to(device)
    # model.load_state_dict(torch.load('Resnest50_0205_1112.mdl'))  #Resnest50_0205_1112
    model.load_state_dict(torch.load('Resnest50_0216_2016.mdl'))
    # print("model",model)

    # Inhosp_Nos,EXAM_NOs, y_trues,y_probs,y_preds,confusionmatrix=Test(model,val_loader,0.9)
    # Inhosp_Nos,EXAM_NOs, y_trues,y_probs,y_preds,confusionmatrix=Train_Test(model,train_loader,0.9)
    Inhosp_Nos, EXAM_NOs, y_trues, y_probs, y_preds, confusionmatrix = Test(model, test_loader, 0.9)

    #     train_AUC=confusion_matrixes(model,train_loader)
    #     print("train_AUC=",train_AUC)
    val_AUC = confusion_matrixes(model, test_loader)
    print("train_AUC=", val_AUC)

    from pandas.core.frame import DataFrame

    c = {"Inhosp_Nos": Inhosp_Nos, "EXAM_NOs": EXAM_NOs, "y_trues": y_trues, "y_probs": y_probs,
         "y_preds": y_preds}  # 将列表a，b转换成字典
    data = DataFrame(c)  # 将字典转换成为数据框
    # data.to_csv("data_train_0214.csv") 使用
    # data.to_csv("data_val_0214.csv")  # 使用
    data.to_csv("data_test_2021_0303.csv")  # 使用
    print("=========文预测结束==============")
    print(data)


