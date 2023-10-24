import sys 
import pandas as pd
import seaborn as sns
import os
import random
import shutil
import sklearn 
import scipy
import numpy as np
import radiomics  #这个库专门用来提取特征
from  radiomics import featureextractor
import SimpleITK as sitk  #读取nii文件
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV#导入Lasso工具包LassoCV
from sklearn.preprocessing import StandardScaler#标准化工具包StandardScaler

path = os.path.abspath('D:/Brad_2019')
sys.path.append(path)
from feature_select import coef, x, y, random_state
from split import hgg_data_test, lgg_data_test


from sklearn.model_selection import train_test_split #分割训练集和验证集
from sklearn.ensemble import RandomForestClassifier #导入随机森林分类器
import joblib #用来保存 sklearn 训练好的模型
#把数据分成训练集和验证集，7：3比例
index_ = coef[coef !=0].index
rforest_x = x[index_]
rforest_y = y
standardscaler = StandardScaler()
rforest_x = standardscaler.fit_transform(rforest_x)#对x进行均值-标准差归一化
x_train,x_test, y_train, y_test = train_test_split(rforest_x,rforest_y,test_size=0.2)
model_forest = RandomForestClassifier(n_estimators=30,random_state=random_state).fit(x_train,y_train)
score = model_forest.score(x_test, y_test)
print("在验证集上的准确率：{}".format(score))
#把随机森林的模型保存下来
joblib.dump(model_forest, 'D:/Brad_2019/model_forest1.model')

import joblib
hgg_test = pd.read_csv('D:/Brad_2019/HGG_test.csv')
lgg_test = pd.read_csv('D:/Brad_2019/LGG_test.csv')
#再把特征值数据和标签数据分开
data_test = pd.concat([hgg_test,lgg_test],axis=0)

x_test_data = data_test[data_test.columns[1:]]
#只提取之前Lasso 筛选后的
index = coef[coef !=0].index
x_test_data = x_test_data[index]

columnNames = x_test_data.columns
x_test_data = x_test_data.astype(np.float32)

x_test_data = standardscaler.transform(x_test_data) #均值-标准差归一化
x_test_data = pd.DataFrame(x_test_data,columns=columnNames)
y_test_data = data_test['label']

print("测试集一共有{}行特征数据，{}列不同特征,包含HGG:{}例,LGG:{}例".format(len(x_test_data),x_test_data.shape[1],len(hgg_data_test),len(lgg_data_test)))
#加载保存后的模型，然后进行预测
model_forest = joblib.load('D:/Brad_2019/model_forest1.model') #这是自己训练模型，记得替换自己的。
score = model_forest.score(x_test_data, y_test_data)
print("在测试集上的准确率：{}".format(score))

from sklearn.metrics import roc_curve, roc_auc_score,auc
kind = {'HGG':1,"LGG":0}
model_forest = joblib.load('D:/Brad_2019/model_forest1.model')#这是自己训练模型，记得替换自己的
label = y_test_data.to_list()  #真实标签
y_predict = model_forest.predict_proba(x_test_data)#得到标签0和1对应的概率
fpr , tpr ,threshold = roc_curve(label, y_predict[:,kind['LGG']], pos_label=kind['LGG'])
roc_auc = auc(fpr,tpr) #计算auc的
print(f'AUC:{roc_auc}')
fpr1 , tpr1 ,threshold = roc_curve(label, y_predict[:,kind['HGG']], pos_label=kind['HGG'])
roc_auc1 = auc(fpr1,tpr1) #计算auc的
plt.plot(fpr, tpr,marker='o', markersize=5,label='LGG')
#plt.plot(fpr1, tpr1,marker='*', markersize=5,label='HGG')
plt.title("LGG AUC:{:.2f}, HGG AUC:{:.2f}".format(roc_auc,roc_auc1))
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc=4)
plt.savefig('D:/Brad_2019/AUC')
plt.close()