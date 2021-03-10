import pandas  as pd
#训练集
data_train_0310=pd.read_csv("data_train_resnest50_2021_0309.csv",index_col=0,converters={"Inhosp_Nos":str})
# print(data_train_0310.head())
# print(data_train_0310.columns)
#
# print(data_train_0310.shape)

data_train_0310_True=data_train_0310[data_train_0310["y_trues"]==data_train_0310["y_preds"]]
data_train_0310_False=data_train_0310[data_train_0310["y_trues"]!=data_train_0310["y_preds"]]
# data_train_0214_True.to_csv("data_train_0214_True.csv")

print(data_train_0310_True.shape)



#
# #验证集
data_val_0310=pd.read_csv("data_val_resnest50_2021_0309.csv",index_col=0,converters={"Inhosp_Nos":str})
data_val_0310_True=data_val_0310[data_val_0310["y_trues"]==data_val_0310["y_preds"]]
data_val_0310_False=data_val_0310[data_val_0310["y_trues"]!=data_val_0310["y_preds"]]

print(data_val_0310_True.shape)

#
#测试集
data_test_0310=pd.read_csv("data_test_resnest50_2021_0309.csv",index_col=0,converters={"Inhosp_Nos":str})
# print(data_test_0310.columns)
data_test_0310_True=data_test_0310[data_test_0310["y_trues"]==data_test_0310["y_preds"]]
data_test_0310_False=data_test_0310[data_test_0310["y_trues"]!=data_test_0310["y_preds"]]



Plane_Data_Predict_True_0310=pd.concat([data_train_0310_True,data_val_0310_True,data_test_0310_True],
                                       axis=0)


Plane_Data_Predict_True_0310=Plane_Data_Predict_True_0310.set_index("Inhosp_Nos")


Plane_Data_Predict_True_0310.to_csv("Plane_Data_Predict_True_0310.csv")
print(len(set(Plane_Data_Predict_True_0310.index)))



# Plane_Data_Predict_True_0310