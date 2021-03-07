import torch
import os, glob
import random, csv
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from PIL import ImageFilter
import re
import pandas  as pd
import numpy  as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

# plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号
# print(os.getcwd())
# images_csv1 = pd.read_csv("Process_Gestational_age1_189.csv")
# print(images_csv1.shape)
#
# images_csv1 = pd.read_csv("Plane/images0805.csv", names=['Inhosp_No', 'img', 'label'],
#                               header=None, converters={0: str})
# images_csv1 = images_csv1.set_index("Inhosp_No")
# print(images_csv1)
# GA=pd.read_csv('Gestational_age_路径索引1.csv', index_col=0,header=None,names=["imgs","GW"])
# # print(GA.columns)
# # print(GA.index)
# GA["Inhosp_No"]=[ item.split("/")[-1].split("_")[0] for item in GA.index]
#
# GA=GA.set_index("Inhosp_No")
#
#
# GA=GA[GA["GW"]<=9.6]
# GA=GA[GA["GW"]>=4.6]
# print(GA)
for name in sorted(os.listdir(os.path.join('Plane')), reverse=False):
    print(name)