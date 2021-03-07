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


def load_csv(root, filename):
    name2label = {}  # "sq...":0

    for name in sorted(os.listdir(os.path.join(root)), reverse=False)[1:]:
        print(name)

        if not os.path.isdir(os.path.join(root, name)):

            print("不是目录", os.path.join(root, name))
            continue
        elif os.path.exists(os.path.join(root, '.DS_Store')):
            print(".DS_Store已存在")
            os.remove(os.path.join(root, '.DS_Store'))

        elif os.path.exists(os.path.join(root, '.ipynb_checkpoints')):
            print(".ipynb_checkpoint存在")
            os.removedirs(os.path.join('./Plane/', '.ipynb_checkpoints'))
        print("name=", name)
        name2label[name] = len(name2label.keys())
    print("name2label", name2label)

    if not os.path.exists(os.path.join(root, filename)):
        print("文件{}不存在".format(filename))
        images = []
        for name in name2label.keys():
            print("name", name)
            # 'pokemon\\mewtwo\\00001.png

            images += glob.glob(os.path.join(root, name, '*.jpg'))

        # 1167, 'pokemon\\bulbasaur\\00000000.png'
        print('len(images)', len(images))

        random.shuffle(images)

        GA = pd.read_csv('Gestational_age_路径索引1.csv', index_col=0, header=None, names=["imgs", "GW"])
        GA["Inhosp_No"] = [item.split("/")[-1].split("_")[0] for item in GA.index]
        GA = GA.set_index("Inhosp_No")
        GA = GA[GA["GW"] <= 9.6]
        GA = GA[GA["GW"] >= 4.6]
        print(GA)

        # Inhosp_Nos = Intersection_标准_不标准切面["0"]
        with open(os.path.join(root, filename), mode='w', newline='') as f:
            writer = csv.writer(f)
            for img in images:
                img1 = img.split("/")[-1].split("_")[0]
                print("******img1********", img1)
                if img1 in list(GA.index):
                    print("******img2********", img1)
                    Inhosp_No = img1
                    EXAM_NO=img.split("/")[-1]
                    name = img.split(os.sep)[-2]
                    # print('name',name)
                    label = name2label[name]
                    # print("label",label)
                    # 'pokemon\\bulbasaur\\00000000.png', 0
                    writer.writerow([Inhosp_No, EXAM_NO,img, label])
            print('writen into csv file:', filename)


    else:

        print("文件{}已经存在".format(filename))


def main():
    load_csv(root='Plane/', filename="images0805.csv")



    images_csv1 = pd.read_csv("Plane/images0805.csv", names=['Inhosp_No', 'EXAM_NO','img', 'label'],
                              header=None, converters={0: str})
    images_csv1 = images_csv1.set_index("Inhosp_No")

    train_val_index = random.sample(list(set(images_csv1.index)), int(0.9 * len(list(set(images_csv1.index)))))
    train_index = random.sample(list(set(train_val_index)), int(0.8 * len(list(set(train_val_index)))))

    val_index = list(set(train_val_index).difference(set(train_index)))

    test_index = list(set(images_csv1.index).difference(set(train_val_index)))

    print("train_val_index", len(train_val_index))
    print("train_index", len(train_index))

    print("val_index", len(val_index))
    print("test_index", len(test_index))

    print("共{}人".format(len(train_index) + len(val_index) + len(test_index)))
    train_data = images_csv1.loc[train_index, :]

    train_data.to_csv("train_data_0307.csv")

    print(train_data.shape)
    val_data = images_csv1.loc[val_index, :]
    val_data.to_csv("val_data_0307.csv")
    print(val_data.shape)
    test_data = images_csv1.loc[test_index, :]
    test_data.to_csv("test_data_0307.csv")
    print(test_data.shape)




if __name__ == '__main__':
    main()
