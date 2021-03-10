
import torch
from Load_data import  Data
import random
from torch import optim, nn
import seaborn as sns
import visdom
import torchvision
from pandas.core.frame import DataFrame
from sklearn import metrics
from resnest.torch import resnest50
from torch.utils.data import DataLoader
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from util import Flatten

import torch
flag = torch.cuda.is_available()
print("正在使用GPU",flag)
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
    #print('val_auc_origin', auc)
    return auc



def Find_Optimal_Cutoff(sensitivities, specificities,preds_all,threshold):
    y = sensitivities + specificities - 1
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    print(Youden_index)
    optimal_threshold = threshold[Youden_index]
    point = [sensitivities[Youden_index], specificities[Youden_index]]
    # accuracy = accuracies[Youden_index]
    # precision=precisions[Youden_index]
    preds_need=preds_all[Youden_index]
    return optimal_threshold,point,preds_need



def Val_optimal_threshold(model, loader):
    model.eval()

    correct = 0
    total = len(loader.dataset)
    Inhosp_Nos_val, EXAM_NOs_val,y_trues_val,y_scores_val = [], [], [], []
    y_probs_val=[]
    for Inhosp_No, EXAM_NO, x, y in loader:
        x, y = x.to(device), y.to(device)

        y_trues_val.extend(y.detach().cpu().numpy().tolist())
        Inhosp_Nos_val.extend(Inhosp_No)
        EXAM_NOs_val.extend(list(EXAM_NO))
        with torch.no_grad():
            logits = model(x)
            y_prob = logits[:, 1]
            y_prob = y_prob.detach().cpu().numpy()
            y_probs_val.extend(list(y_prob))

            y_scores_val.append(logits)


    y_trues_one_hot = torch.tensor([i for i in y_trues_val])

    y_trues_one_hot = to_categorical(y_trues_one_hot, 2)

    y_scores_val = torch.cat([i for i in y_scores_val], 0)

    y_scores_val = y_scores_val.cpu().numpy()

    auc = roc_auc_score(y_trues_one_hot, y_scores_val)
    print("val_AUC=", auc)
    # print('print("val_AUC=", val_AUC)',auc)

   # print("y_probs",y_probs)
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sensitivities=[]
    specificities = []
    accuracies=[]
    precisions=[]
    preds_all=[]
    for i, item in enumerate(thresholds):
        # Yuedens=0
        print('第{}th thresholds{}'.format(i, item))

        preds = []
        for j in y_probs_val:

            if j > item:
                preds.append(1)

            else:
                preds.append(0)
        preds_all.append(preds)
        confusion_matrix0 = confusion_matrix(y_trues_val, preds)
        print(confusion_matrix0)
        sensitivity = confusion_matrix0[1, 1] / (confusion_matrix0[1, 0] + confusion_matrix0[1, 1])
        specificity = confusion_matrix0[0, 0] / (confusion_matrix0[0, 0] + confusion_matrix0[0, 1])
        # accuracy=(confusion_matrix0[0, 0] +confusion_matrix0[1, 1])/(confusion_matrix0[0, 0] +confusion_matrix0[1, 1]+
        #                                                              confusion_matrix0[1, 0]+confusion_matrix0[0, 1])
        # precision=confusion_matrix0[1, 1]/(confusion_matrix0[1, 1]+confusion_matrix0[0, 1])

        sensitivities.append(sensitivity)
        specificities.append(specificity)
        # accuracies.append(accuracy)
        # precisions.append(precision)
    sensitivities = np.array(sensitivities)
    specificities = np.array(specificities)
    # accuracies=np.array(accuracies)
    # precisions = np.array(precisions)




    optimal_threshold,point,preds_need =Find_Optimal_Cutoff(sensitivities, specificities,preds_all,thresholds)

    print('optimal_threshold=', optimal_threshold)
    print("point=", point)

    return optimal_threshold,point,Inhosp_Nos_val,EXAM_NOs_val,y_trues_val,y_probs_val,preds_need

# Inhosp_Nos, EXAM_NOs, y_trues, y_probs, preds, confusionmatrix


# def val_Test(model, loader, optimal_threshold):
#     model.eval()
#
#     correct = 0
#     total = len(loader.dataset)
#     Inhosp_Nos, EXAM_NOs, y_trues, y_probs, preds = [], [], [], [], []
#     for Inhosp_No, EXAM_NO, x, y in loader:
#         x, y = x.to(device), y.to(device)
#         y_trues.extend(list(y.cpu().numpy()))
#         Inhosp_Nos.extend(Inhosp_No)
#         # print("EXAM_NO",list(EXAM_NO))
#         EXAM_NOs.extend(list(EXAM_NO))
#
#         with torch.no_grad():
#             logits = model(x)
#             y_prob = logits[:, 1]
#             y_prob = y_prob.detach().cpu().numpy()
#
#             y_probs.extend(list(y_prob))
#
#     for j in y_probs:
#
#         if j > optimal_threshold:
#             preds.append(1)
#
#         else:
#             preds.append(0)
#
#     confusionmatrix = confusion_matrix(y_trues, preds)
#     print('confusionmatrix_val', confusionmatrix)
#     # pred = logits.argmax(dim=1)
#
#     #             y_preds.extend(list(pred.detach().cpu().numpy()))
#
#     return Inhosp_Nos, EXAM_NOs, y_trues, y_probs, preds, confusionmatrix
#
#

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
    print('confusionmatrix_train', confusionmatrix)
    # pred = logits.argmax(dim=1)

    #             y_preds.extend(list(pred.detach().cpu().numpy()))

    return Inhosp_Nos, EXAM_NOs, y_trues, y_probs, y_preds, confusionmatrix




def Test(model, loader, optimal_threshold):
    model.eval()

    correct = 0
    total = len(loader.dataset)
    Inhosp_Nos, EXAM_NOs, y_trues, y_probs, preds = [], [], [], [], []
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
            preds.append(1)

        else:
            preds.append(0)

    confusionmatrix = confusion_matrix(y_trues, preds)
    print('confusionmatrix_test', confusionmatrix)
    # pred = logits.argmax(dim=1)

    #             y_preds.extend(list(pred.detach().cpu().numpy()))

    return Inhosp_Nos, EXAM_NOs, y_trues, y_probs, preds, confusionmatrix









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



    print("*******************计算验证集的混淆矩阵*****************************")
    optimal_threshold,point,Inhosp_Nos_val,EXAM_NOs_val,y_trues_val,y_probs_val,preds_need= Val_optimal_threshold(model, val_loader)
    # val_AUC = confusion_matrixes(model, val_loader)

    # print("optimal_threshold=",optimal_threshold)
    # print("point=", point)
    print("验证集样本预测中....")
    c = {"Inhosp_Nos": Inhosp_Nos_val, "EXAM_NOs": EXAM_NOs_val, "y_trues": y_trues_val,
         "y_probs": y_probs_val,
         "y_preds": preds_need}  # 将列表a，b转换成字典
    data = DataFrame(c)  # 将字典转换成为数据框
    data.to_csv("data_val_resnest50_2021_0309.csv")  # 使用
    print("=========验证集所有样本预测结束==============")

    print("*********样本筛选中******************")

    ###训练集的样本预测开始
    Inhosp_Nos_train, EXAM_NOs_train, y_trues_train, y_probs_train, y_preds_train, confusionmatrix_train=Train_Test(model, train_loader,optimal_threshold)
    train_AUC = Train_confusion_matrixes(model, train_loader)
    print("train_AUC=", train_AUC)

    c = {"Inhosp_Nos": Inhosp_Nos_train, "EXAM_NOs": EXAM_NOs_train, "y_trues": y_trues_train, "y_probs":y_probs_train,
         "y_preds": y_preds_train}  # 将列表a，b转换成字典
    data = DataFrame(c)  # 将字典转换成为数据框
    # data.to_csv("data_train_0214.csv") 使用
    # data.to_csv("data_val_0214.csv")  # 使用
    data.to_csv("data_train_resnest50_2021_0309.csv")  # 使用
    print("=========训练集所有样本预测结束==============")


    print("*********计算测试集的预测结果******************")
    Inhosp_Nos_test, EXAM_NOs_test, y_trues_test, y_probs_test, y_preds_test, confusionmatrix_test= Test(model, test_loader,optimal_threshold)
    test_AUC = confusion_matrixes(model, test_loader)
    print("test_AUC=", test_AUC)

    c = {"Inhosp_Nos": Inhosp_Nos_test, "EXAM_NOs": EXAM_NOs_test, "y_trues": y_trues_test,
         "y_probs": y_probs_test,"y_preds": y_preds_test}  # 将列表a，b转换成字典
    data = DataFrame(c)  # 将字典转换成为数据框
    data.to_csv("data_test_resnest50_2021_0309.csv")  # 使用

    print("**********End**********")







if __name__ == '__main__':
    main()








