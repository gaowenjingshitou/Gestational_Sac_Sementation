import torch
import os, glob
import random, csv
import pandas as pd
import numpy as np



Plane_Data_Predict_True_0310=pd.read_csv("Plane_Data_Predict_True_0310.csv",converters={"Inhosp_Nos":str})
Plane_Data_Predict_True_0310=Plane_Data_Predict_True_0310.set_index("Inhosp_Nos")
Standard_Plane=Plane_Data_Predict_True_0310[Plane_Data_Predict_True_0310["y_trues"]==1]
Non_Standard_Plane=Plane_Data_Predict_True_0310[Plane_Data_Predict_True_0310["y_trues"]==0]




Intersection=list(set(Standard_Plane.index).intersection(set(Non_Standard_Plane.index)))
Intersection_标准_不标准=pd.DataFrame(Intersection)
Intersection_标准_不标准.to_csv("Intersection_标准_不标准切面_After_Choose_Sample_2021_0310.csv")


print(len(Intersection))  #167



#
Plane_Data_Predict_True_0310_Choose=Plane_Data_Predict_True_0310.loc[Intersection,:]   #[6442 rows x 4 columns]
Plane_Data_Predict_True_0310_Choose.to_csv("Plane_Data_Predict_True_0310_Choose.csv")

print(Plane_Data_Predict_True_0310_Choose)



# images=[]

# images=os.listdir("/workspace/标准平面的选取算法/Plane/Non_Standard_Plane/")
# print(len(images))
# images=[ item.split(".")[0].split("_")[0] for item in images]
# print(len(images))
#
# images1=[]
# images1 += glob.glob(os.path.join('./Plane/Standard_plane/',  '*.jpg'))
# print(len(images1))
# images1=[ item.split("/")[-1].split("_")[0] for item in images1]
# print(len(images1))
# Intersection=list(set(images).intersection(set(images1)))
# Intersection_标准_不标准=pd.DataFrame(Intersection)
# Intersection_标准_不标准.to_csv("Intersection_标准_不标准切面_After_Choose_Sample.csv")
# Intersection_标准_不标准