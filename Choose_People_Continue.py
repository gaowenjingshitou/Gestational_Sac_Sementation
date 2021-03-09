import pandas  as pd
#训练集
data_train_0214=pd.read_csv("data_train_0214.csv",index_col=0,converters={"Inhosp_Nos":str})
data_train_0214_True=data_train_0214[data_train_0214["y_trues"]==data_train_0214["y_preds"]]
data_train_0214_False=data_train_0214[data_train_0214["y_trues"]!=data_train_0214["y_preds"]]
# data_train_0214_True.to_csv("data_train_0214_True.csv")


#验证集
data_val_0214=pd.read_csv("data_val_0214.csv",index_col=0,converters={"Inhosp_Nos":str})
data_val_0214_True=data_val_0214[data_val_0214["y_trues"]==data_val_0214["y_preds"]]
data_val_0214_False=data_val_0214[data_val_0214["y_trues"]!=data_val_0214["y_preds"]]




#测试集
data_test_0214=pd.read_csv("data_test_0214.csv",index_col=0,converters={"Inhosp_Nos":str})
data_test_0214_True=data_test_0214[data_test_0214["y_trues"]==data_test_0214["y_preds"]]

data_test_0214_False=data_test_0214[data_test_0214["y_trues"]!=data_test_0214["y_preds"]]


