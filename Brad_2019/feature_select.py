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

#通过T检验从106个特征筛选
from scipy.stats import levene, ttest_ind

path = os.path.abspath('D:/Brad_2019/split.py')
sys.path.append(path)
from split import hgg_data_train, lgg_data_train, random_state


counts = 0
columns_index =[]
for column_name in hgg_data_train.columns[1:]:
    if levene(hgg_data_train[column_name], lgg_data_train[column_name])[1] > 0.05:
        if ttest_ind(hgg_data_train[column_name],lgg_data_train[column_name],equal_var=True)[1] < 0.05:
            columns_index.append(column_name)
    else:
        if ttest_ind(hgg_data_train[column_name],lgg_data_train[column_name],equal_var=False)[1] < 0.05:
            columns_index.append(column_name)

print("筛选后剩下的特征数：{}个".format(len(columns_index)))

from kydavra import MUSESelector,PointBiserialCorrSelector,LassoSelector,ChiSquaredSelector,PointBiserialCorrSelector

#数据只保留从T检验筛选出的特征数据，重新组合成data
if  not 'label' in columns_index:
    columns_index = ['label'] + columns_index
hgg_train = hgg_data_train[columns_index]  
lgg_train = lgg_data_train[columns_index]  

data = pd.concat([hgg_train, lgg_train])
data = data.sample(frac=1.0,random_state=random_state)  # 全部打乱

#缪斯选择器筛选特征
#主要思想是在一个特征下，不同 类别的分布是有明显差异的，如果各个类别都是均匀分布，那这个特征就没有用。
max_columns_num = 20  #这个值是人工定义值
muse = MUSESelector (num_features=max_columns_num)
columns_index = muse.select(data, 'label')

print("筛选后剩下的特征数：{}个".format(len(columns_index)))


#数据只保留从T检验筛选出的特征数据，重新组合成data
if  not 'label' in columns_index:
    columns_index = ['label'] + columns_index
hgg_train = hgg_data_train[columns_index]  
lgg_train = lgg_data_train[columns_index]  

data = pd.concat([hgg_train, lgg_train])
data = data.sample(frac=1.0,random_state=random_state)  # 全部打乱

#再把特征值数据和标签数据分开
x = data[data.columns[1:]]
y = data['label']
#先保存X的列名
columnNames = x.columns

lassoCV_x = x.astype(np.float64)#把x数据转换成np.float格式
lassoCV_y = y.astype(np.float64)

standardscaler = StandardScaler()
lassoCV_x = standardscaler.fit_transform(lassoCV_x)#对x进行均值-标准差归一化
lassoCV_x = pd.DataFrame(lassoCV_x,columns=columnNames)#转 DataFrame 格式

# 形成5为底的指数函数
# 5**（-3） ~  5**（-2）
alpha_range = np.logspace(-3,-2,50,base=5)
#alpha_range在这个参数范围里挑出aplpha进行训练，cv是把数据集分5分，进行交叉验证，max_iter是训练1000轮
lassoCV_model = LassoCV(alphas=alpha_range,cv=5,max_iter=1000)
#进行训练
lassoCV_model.fit(lassoCV_x,lassoCV_y)

#打印训练找出来的入值
print(lassoCV_model.alpha_)
# print("Coefficient of the model:{}".format(lassoCV_model.coef_) )
# print("intercept of the model:{}".format(lassoCV_model.intercept_))

coef = pd.Series(lassoCV_model.coef_, index=columnNames)
print("从原来{}个特征，筛选剩下{}个".format(len(columnNames),sum(coef !=0)))
print("分别是以下特征")
print(coef[coef !=0])
index = coef[coef !=0].index
lassoCV_x = lassoCV_x[index]
# lassoCV_x.head()

import seaborn as sns
f, ax= plt.subplots(figsize = (10, 10))
sns.heatmap(lassoCV_x.corr(),annot=True,cmap='coolwarm',annot_kws={'size':10,'weight':'bold', },ax=ax)#绘制混淆矩阵
ax.set_xticklabels(ax.get_xticklabels(), rotation=45,va="top",ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
plt.savefig('D:/Brad_2019/corr')
plt.close()