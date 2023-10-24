import pandas as pd
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
import seaborn as sns
#%matplotlib inline

#重新制作Mask标签
def createNewMask(file_path,dst_path):
    file_base_name = os.path.basename(file_path)
    sitkImage = sitk.ReadImage(file_path)          #读取nii.gz文件
    npImage = sitk.GetArrayFromImage(sitkImage)    #simpleITK 转换成numpy
    npImage[npImage > 0 ] =1                       #把大于0的标签都变成1，就是所有病区都要
    outImage = sitk.GetImageFromArray(npImage)     #numpy 转换成simpleITK
    outImage.SetSpacing(sitkImage.GetSpacing())    #设置和原来nii.gz文件一样的像素空间
    outImage.SetOrigin(sitkImage.GetOrigin())      #设置和原来nii.gz文件一样的原点位置
    sitk.WriteImage(outImage,os.path.join(dst_path,file_base_name))#保存文件

if not os.path.exists('D:/data/MICCAI_BraTS_2019_Data/MyData'):
    os.makedirs('D:/data/MICCAI_BraTS_2019_Data/MyData')

source_data_path = 'D:/data/MICCAI_BraTS_2019_Data/MICCAI_BraTS_2019_Data_Training'
dst_data_path = 'D:/data/MICCAI_BraTS_2019_Data/MyData'
kinds = ['HGG','LGG']
index = 0
for kind in kinds:
    kind_path = os.path.join(source_data_path, kind)    #/home/aistuido/work/MICCAI_BraTS_2019_Data_Training/HGG
    for folder in os.listdir(kind_path):
        file_dir_path =  os.path.join(kind_path,folder )#/home/aistuido/work/MICCAI_BraTS_2019_Data_Training/HGG/BraTS19_CBICA_APY_1
        dst_path =  os.path.join(dst_data_path, kind,str(index)) #/home/aistuido/work/MyData/HGG/1
        for file_name in os.listdir(file_dir_path):
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            if 't1ce' in file_name:
                dst_file_path = os.path.join(dst_path, file_name)#/home/aistuido/work/MyData/HGG/1/BraTS19_2013_5_1_t1ce.nii.gz
                shutil.copy(os.path.join(file_dir_path,file_name),dst_file_path)
                index += 1
            elif 'seg' in file_name:
                createNewMask(os.path.join(file_dir_path,file_name), dst_path)
print("完成")

#HGG.csv 和LGG.csv文件

kinds = ['HGG','LGG']
para_path = 'D:/data/MICCAI_BraTS_2019_Data/MR_1mm.yml'#这个是特征处理配置文件，具体可以参考pyradiomics官网网站
extractor = featureextractor.RadiomicsFeatureExtractor(para_path) 
for kind in kinds:
    print("{}:开始提取特征".format(kind))
    features_dict = dict()
    df = pd.DataFrame()
    path = 'D:/data/MICCAI_BraTS_2019_Data/MyData/' + kind
    # 使用配置文件初始化特征抽取器
    for index, folder in enumerate( os.listdir(path)):
        for f in os.listdir(os.path.join(path, folder)):
            if 't1ce' in f:
                ori_path = os.path.join(path, folder, f)
            else:
                lab_path = os.path.join(path, folder, f)
        features = extractor.execute(ori_path,lab_path)  #抽取特征
        for key, value in features.items():  #输出特征
            if 'diagnostics_Versions' in  key or 'diagnostics_Configuration' in key:#这些都是一些共有的特征，可以去掉
                continue
            features_dict[key] = value
        df = df.append(pd.DataFrame.from_dict(features_dict.values()).T,ignore_index=True)
        print(index)
    df.columns = features_dict.keys()
    df.to_csv('{}.csv'.format(kind),index=0)
print("完成")