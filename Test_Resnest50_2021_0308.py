
import torch
from Load_data import  Data

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


def Train_Test(model, loader):
    model.train()

    correct = 0
    total = len(loader.dataset)
    Inhosp_Nos, y_trues, y_probs, y_preds = [], [], [], []
    for Inhosp_No, EXAM_NO, x, y in loader:
        x, y = x.to(device), y.to(device)
        y_trues.extend(list(y.cpu().numpy()))
        Inhosp_Nos.extend(Inhosp_No)
        with torch.no_grad():
            logits = model(x)
            y_prob = logits[:, 1]
            y_prob = y_prob.detach().cpu().numpy()
            y_probs.extend(list(y_prob))

            pred = logits.argmax(dim=1)
            y_preds.extend(list(pred.detach().cpu().numpy()))

    return Inhosp_Nos, y_trues, y_probs, y_preds


def Test(model, loader):
    model.eval()

    correct = 0
    total = len(loader.dataset)
    Inhosp_Nos, y_trues, y_probs, y_preds = [], [], [], []
    for Inhosp_No, EXAM_NO, x, y in loader:
        x, y = x.to(device), y.to(device)
        y_trues.extend(list(y.cpu().numpy()))
        Inhosp_Nos.extend(Inhosp_No)
        with torch.no_grad():
            logits = model(x)
            y_prob = logits[:, 1]
            y_prob = y_prob.detach().cpu().numpy()
            y_probs.extend(list(y_prob))

            pred = logits.argmax(dim=1)
            y_preds.extend(list(pred.detach().cpu().numpy()))

    return Inhosp_Nos, y_trues, y_probs, y_preds


def Find_Optimal_Cutoff(sensitivities, specificities, threshold):
    y = sensitivities + specificities - 1
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    print(Youden_index)
    optimal_threshold = threshold[Youden_index]
    point = [sensitivities[Youden_index], specificities[Youden_index]]
    return optimal_threshold, point


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
    model.load_state_dict(torch.load('Resnest50_0308.mdl'))

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print("*******************计算验证集的混淆矩阵*****************************")
    Inhosp_Nos_val, y_trues_val, y_probs_val, y_preds_val = Test(model, val_loader)

    sensitivities = []
    specificities = []
    for i, item in enumerate(thresholds):
        # Yuedens=0
        print('第{}th thresholds{}'.format(i, item))

        preds = []
        for j in y_probs_val:

            if j > item:
                preds.append(1)

            else:
                preds.append(0)
        confusion_matrix0 = confusion_matrix(y_trues_val, preds)
        print(confusion_matrix0)
        sensitivity = confusion_matrix0[1, 1] / (confusion_matrix0[1, 0] + confusion_matrix0[1, 1])
        specificity = confusion_matrix0[0, 0] / (confusion_matrix0[0, 0] + confusion_matrix0[0, 1])

        sensitivities.append(sensitivity)
        specificities.append(specificity)
    sensitivities = np.array(sensitivities)
    specificities = np.array(specificities)
    optimal_threshold, point = Find_Optimal_Cutoff(sensitivities, specificities, thresholds)

    print('optimal_threshold=', optimal_threshold)
    print("point=", point)

    preds_val = []
    for j in y_probs_val:

        if j > optimal_threshold:
            preds_val.append(1)

        else:
            preds_val.append(0)

    val_AUC = confusion_matrixes(model, val_loader)
    print("val_AUC=", val_AUC)

    confusionmatrix_val = confusion_matrix(y_trues_val, preds_val)
    print('confusionmatrix_val', confusionmatrix_val)

    print("*********计算训练集的混淆矩阵******************")
    ###训练集的混淆矩阵
    Inhosp_Nos_train, y_trues_train, y_probs_train, y_preds_train = Train_Test(model, train_loader)

    preds_train = []
    for j in y_probs_train:

        if j > optimal_threshold:
            preds_train.append(1)

        else:
            preds_train.append(0)

    confusionmatrix_train = confusion_matrix(y_trues_train, preds_train)

    print('Confusionmatrix_train', confusionmatrix_train)

    train_AUC = Train_confusion_matrixes(model, train_loader)
    print("train_AUC=", train_AUC)

    print("*********计算测试集的混淆矩阵******************")

    test_AUC = confusion_matrixes(model, test_loader)
    print("test_AUC=", test_AUC)

    ###训练集的混淆矩阵
    Inhosp_Nos_test, y_trues_test, y_probs_test, y_preds_test = Test(model, test_loader)

    preds_test = []
    for j in y_probs_test:

        if j > optimal_threshold:
            preds_test.append(1)

        else:
            preds_test.append(0)

    confusionmatrix_test = confusion_matrix(y_trues_test, preds_test)
    print('confusionmatrix_test', confusionmatrix_test)


if __name__ == '__main__':
    main()








