import torch
import os, glob
from  torch.utils.data import Dataset , DataLoader
from torchvision import transforms
from PIL import Image
from PIL import ImageFilter
import re
import pandas  as pd
import numpy  as np
import random


class Data(Dataset):

    def __init__(self, root, resize, mode):
        super(Data, self).__init__()

        self.root = root
        self.resize = resize

        self.name2label = {}  # "sq...":0

        for name in sorted(os.listdir(os.path.join(root)), reverse=False):
            print(name)
            if not os.path.isdir(os.path.join(root, name)):
                print("不是目录", os.path.join(root, name))
                continue
            elif os.path.exists(os.path.join(root, '.ipynb_checkpoints')):
                os.removedirs(os.path.join(root, '.ipynb_checkpoints'))
            self.name2label[name] = len(self.name2label.keys())
        print("name2label", self.name2label)  # .ipynb_checkpoints [0:2]:  #['Non_Standard_Plane', 'Standard_plane']

        # 训练集
        train_data = pd.read_csv("train_data_0310.csv", converters={"Inhosp_No": str})
        train_data = train_data.set_index("Inhosp_No")
        EXAM_NO = train_data["img"]
        EXAM_NO = [item.split("/")[-1] for item in EXAM_NO]
        train_data["EXAM_NO"] = EXAM_NO

        # 验证集
        val_data = pd.read_csv("val_data_0310.csv", converters={"Inhosp_No": str})  # 312 rows × 3 columns
        val_data = val_data.set_index("Inhosp_No")

        EXAM_NO = val_data["img"]
        EXAM_NO = [item.split("/")[-1] for item in EXAM_NO]
        val_data["EXAM_NO"] = EXAM_NO

        # 测试集
        test_data = pd.read_csv("test_data_0310.csv", converters={"Inhosp_No": str})  # 312 rows × 3 columns
        test_data = test_data.set_index("Inhosp_No")
        EXAM_NO = test_data["img"]
        EXAM_NO = [item.split("/")[-1] for item in EXAM_NO]
        test_data["EXAM_NO"] = EXAM_NO

        train_val_data = pd.concat([train_data, val_data], axis=0)
        train_val_data_index = np.array(list(set(train_val_data.index)))

        if mode == "train":
            print("====Enter {} mode=====".format(mode))
            print("训练集有{}数据".format(train_data.shape))
            print("训练集有{}人".format(len(set(train_data.index))))

            self.Inhosp_Nos = train_data.index
            self.EXAM_NOs = train_data["EXAM_NO"]
            self.imgs = train_data["img"]

            self.labels = train_data["label"]
            print
        elif mode == "val":
            print("====Enter {} mode=====".format(mode))
            print("验证集集有{}数据".format(val_data.shape))
            print("验证集集有{}人".format(len(set(val_data.index))))
            self.Inhosp_Nos = val_data.index
            self.EXAM_NOs = val_data["EXAM_NO"]
            self.imgs = val_data["img"]
            self.labels = val_data["label"]


        else:
            print("====Enter test mode=====")
            print("测试集有{}数据".format(test_data.shape))
            print("测试集有{}人".format(len(set(test_data.index))))
            self.Inhosp_Nos = test_data.index
            self.EXAM_NOs = test_data["EXAM_NO"]
            self.imgs = test_data["img"]
            self.labels = test_data["label"]

    def __len__(self):

        return len(self.imgs)

    def denormalize(self, x_hat):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # x_hat = (x-mean)/std
        # x = x_hat*std = mean
        # x: [c, h, w]
        # mean: [3] => [3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        # print(mean.shape, std.shape)
        x = x_hat * std + mean

        return x

    def __getitem__(self, idx):

        # print("self.Inhosp_No",len(self.Inhosp_No))
        # print('self.EXAM_NO',len(self.EXAM_NO))
        Inhosp_No, EXAM_NO, img, label = self.Inhosp_Nos[idx], self.EXAM_NOs[idx], self.imgs[idx], self.labels[idx]

        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),  # string path= > image data
            # lambda x:x.filter(ImageFilter.SMOOTH),
            # transforms.RandomRotation(30, resample=False, expand=False, center=None),
            # transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0),

            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        img = tf(img)
        label = torch.tensor(label)

        return Inhosp_No, EXAM_NO, img, label

def main():
    import visdom
    import time
    import torchvision

    viz = visdom.Visdom()

    train = Data('Plane', 224, 'val')

    #     db1 = Data('/workspace/NEW_DATA_0914/', 224, 'val')
    #     db2 = Data('/workspace/NEW_DATA_0914/', 224, 'test')
    Inhosp_No, EXAM_NO, x, y = next(iter(train))

    print("Inhosp_No", Inhosp_No, "EXAM_NO", EXAM_NO, 'sample:', x.shape, y.shape, y)

    viz.image(train.denormalize(x), win='sample_x', opts=dict(title='sample_x'))

    train_fold1_loader = DataLoader(train, batch_size=32, shuffle=True, num_workers=0)

    for Inhosp_No, EXAM_NO, x, y in train_fold1_loader:
        print("batch sample", x.shape, y.shape)
        viz.images(train.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
        viz.text(str(Inhosp_No), win='Inhosp_No', opts=dict(title='batch-Inhosp_No'))
        viz.text(str(EXAM_NO), win='EXAM_NO', opts=dict(title='batch-EXAM_NO'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))

        time.sleep(10)


if __name__ == '__main__':
    main()