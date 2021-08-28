import pandas as pd
import numpy as np
import scipy, sklearn, openpyxl, xlwt
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def Preprocess():
    ### Load Dataset
    data = pd.read_excel('BrainAgeData.xlsx', engine = 'openpyxl',sheet_name=0)
    data = data.drop(['Unnamed: 0'],axis=1, inplace=False)

    ### Preprocess Data
    # y(target) 분리
    y_target = data['age']

    # X(features) 분리 - 성별을 하나의 categorical feature로 두고 회귀 진행
    X_features = data.drop('age', axis=1, inplace=False)

    # atlas1 만 사용                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       만 사용
    atlas_1 = X_features.iloc[:,0:449]

    # 관련 features(A+B+C+D)
    abcd = X_features.iloc[:,657:743]

    # 사용할 데이터 묶기
    atlas_1abcd = pd.concat([atlas_1, abcd],axis=1)


    ### Scale Features
    # StandardScaler
    scaler = StandardScaler()   # mean:0, variance:1
    scaler.fit(atlas_1abcd)
    scaled = scaler.transform(atlas_1abcd)
    scaled_1abcd = pd.DataFrame(data=scaled, columns= atlas_1abcd.columns)
    
    return scaled_1abcd, y_target



