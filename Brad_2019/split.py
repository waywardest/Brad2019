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


#读取HGG.csv和LGG.csv ，因为是分类任务，新增标签0是LGG，1标签是HGG
#还有各从HGG.csv 和LGG.csv随机抽取20%作为测试集

def split_df(df, ratio):
    #用来分割数据集，保留一定比例的数据集当做最终的测试集
    cut_idx = int(round(ratio * df.shape[0]))
    data_test, data_train = df.iloc[:cut_idx], df.iloc[cut_idx:]
    return (data_train, data_test)

test_ratio = 0.2
random_state = 2021 #固定随机种子
hgg_data = pd.read_csv('D:/Brad_2019/HGG.csv')
lgg_data = pd.read_csv('D:/Brad_2019/LGG.csv')

hgg_data.insert(0,'label', 1) #插入标签
lgg_data.insert(0,'label', 0) #插入标签

hgg_data = hgg_data.sample(frac=1.0, random_state=random_state)  # 全部打乱
lgg_data = lgg_data.sample(frac=1.0, random_state=random_state)  # 全部打乱

#因为有些特征是字符串，直接删掉
cols=[x for i,x in enumerate(hgg_data.columns) if type(hgg_data.iat[1,i]) == str]
hgg_data=hgg_data.drop(cols,axis=1)
cols=[x for i,x in enumerate(lgg_data.columns) if type(lgg_data.iat[1,i]) == str]
lgg_data=lgg_data.drop(cols,axis=1)

hgg_data_train, hgg_data_test = split_df(hgg_data,test_ratio) #返回train 和test数据集
lgg_data_train, lgg_data_test = split_df(lgg_data,test_ratio) #返回train 和test数据集

#保存测试集为cvs 后面最终验证使用
hgg_data_test.to_csv('D:/Brad_2019/HGG_test.csv',index=False)
lgg_data_test.to_csv('D:/Brad_2019/LGG_test.csv',index=False)

# 查看总数据类别是否平衡
fig, ax = plt.subplots()
sns.set()
total_data = pd.concat([hgg_data, lgg_data])
ax = sns.countplot(x='label',hue='label',data=total_data)
print(total_data['label'].value_counts())
plt.savefig('D:/Brad_2019/total')
plt.close()
#把hgg_data_train 和lgg_data_train 并在一起并且打乱。
#查看总体数据情况
data = pd.concat([hgg_data_train, lgg_data_train])
data = data.sample(frac=1.0,random_state=random_state)  # 全部打乱
print("一共有{}行特征数据".format(len(data)))
print("一共有{}列不同特征".format(data.shape[1]))
#再把特征值数据和标签数据分开
x = data[data.columns[1:]]
y = data['label']
#取X的5行看看数据
# x.head()