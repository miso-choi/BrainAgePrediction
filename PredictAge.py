from data_preprocessing import Preprocess
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
            import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared, ConstantKernel
import warnings
warnings.filterwarnings(action='ignore')

def LoadTest():
    test_set = pd.read_excel('./ExternalValidation.xlsx', engine='openpyxl')
    test_set.drop(['Unnamed: 0'],axis=1, inplace=True)
    
    ### Train Data와 동일한 조건을 갖도록 Preprocessing
    # Destrieux's atlas (atlas1)
    atlas_1 = test_set.iloc[:,0:449]

    # region A+B+C+D
    abcd = test_set.iloc[:,657:743]

    # destrieux(atlas1) + abcd
    atlas_1abcd = pd.concat([atlas_1, abcd],axis=1)

    ## StandardScaler
    scaler = StandardScaler()   # 평균 0, 분산 1
    scaler.fit(atlas_1abcd)
    scaled = scaler.transform(atlas_1abcd)
    scaled_test = pd.DataFrame(data=scaled, columns= atlas_1abcd.columns)
    
    return scaled_test


    
if __name__ == "__main__":
    ## Load Train data   - shape : (2184, 535)
    scaled_1abcd, y_target = Preprocess()
    
    ## Load Test data   - shape : (50, 535)
    scaled_test = LoadTest()
        
    
    ## GaussianProcessRegression 의 경우
    # 모델 선언
    sigma=21; l1 = 45; coef = 25; Alpha = 0.4; l2 = 10
    k2 = sigma**2 * RBF(length_scale=l1) + coef*RationalQuadratic(length_scale = l2, alpha = Alpha)
    gpr = GaussianProcessRegressor(kernel = k2,random_state = 128, n_restarts_optimizer=10,normalize_y=True)
    
    # 모델.fit
    gpr.fit(scaled_1abcd, y_target)
    
    # 모델.predict
    pred = gpr.predict(scaled_test)
    pred_df = pd.DataFrame(pred, columns = ['predicted age'])
    print(pred_df)
    
    
    '''
    ## ElasticNet, RandomForest, LightGBM 의 경우
    # 모델 선언 (각 모델에서 반환된 best_model의 parameter로)
    enet = ElasticNet(alpha=0.1, l1_ratio=0.4)
    
    # 모델.fit
    enet.fit(scaled_1abcd, y_target)
    
    # 모델.predict
    pred = enet.predict(scaled_test)
    pred_df = pd.DataFrame(pred, columns = ['predicted age'])
    print(pred_df)
    '''