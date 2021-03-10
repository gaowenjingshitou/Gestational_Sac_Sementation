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

Plane_Data_Predict_True_0310_Choose=Plane_Data_Predict_True_0310.loc[Intersection,:]   #[6442 rows x 4 columns]
Plane_Data_Predict_True_0310_Choose.to_csv("Plane_Data_Predict_True_0310_Choose.csv")

print(Plane_Data_Predict_True_0310_Choose.shape)
