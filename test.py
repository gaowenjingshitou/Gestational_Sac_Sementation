
import  pandas as pd
import os
import numpy as np

images_csv1 = pd.read_csv("Plane/images0805.csv", names=['Inhosp_No', 'img', 'label'],
                          header=None, converters={0: str})
images_csv1 = images_csv1.set_index("Inhosp_No")
print(images_csv1.shape)


S=images_csv1[images_csv1["label"]==1]
N=images_csv1[images_csv1["label"]==0]

print("标准切面图像=",S.shape)
print("标准切面图像的人数",len(set(S.index)))
print("非标准切面图像",N.shape)
print("非标准切面图像的人数",len(set(N.index)))