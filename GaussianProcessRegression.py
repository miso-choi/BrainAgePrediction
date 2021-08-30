from data_preprocessing import Preprocess
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
            import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared, ConstantKernel, WhiteKernel
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from tqdm import tqdm


def TrainGPR():
    n_iter = 0
    kfold = KFold(n_splits=10, shuffle=True)
    kernel_info = []
    cv_test_mae = []
    cv_train_mae = []

    # KFold객체의 split( ) 호출하면 폴드 별 학습용, 검증용 데이터의 로우 인덱스를 array로 반환  
    for train_index, test_index in tqdm(kfold.split(np.array(scaled_1abcd))):
        # kfold.split( )으로 반환된 인덱스를 이용하여 학습용, 검증용 데이터 추출
        X_train, X_test = np.array(scaled_1abcd)[train_index], np.array(scaled_1abcd)[test_index]
        y_train, y_test = np.array(y_target)[train_index], np.array(y_target)[test_index]
        
        #학습 및 예측 
        gpr.fit(X_train,y_train)
        y_test_pred = gpr.predict(X_test)
        y_train_pred = gpr.predict(X_train)
        
        n_iter += 1  # CV 교차검증 (1~10 print)

        ###GetMAE()
        # CV 반복 마다 정확도 측정 
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)

        print('\n#{0} 교차 검증 test MAE :{1}, train MAE :{2}'
              .format(n_iter, test_mae, train_mae))
        print("Learned kernel: %s \n" % gpr.kernel_)
        
        kernel_info.append(gpr.kernel_) 
        cv_test_mae.append(test_mae)
        cv_train_mae.append(train_mae)

    # 개별 iteration별 정확도를 합하여 평균 정확도 계산 
    mean_test_mae = np.mean(cv_test_mae)
    mean_train_mae = np.mean(cv_train_mae)
        
    return kernel_info, mean_test_mae, mean_train_mae


if __name__ == "__main__":
    scaled_1abcd, y_target = Preprocess()

    sigma = 21; l_1 = 45; coef = 25; a = 0.4; l_2 = 10 
    k = sigma**2 * RBF(length_scale = l_1) + coef*RationalQuadratic(length_scale = l_2, alpha = a) + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel = k,random_state = 128, n_restarts_optimizer=10, normalize_y=True)

    kernel_info, mean_test_mae, mean_train_mae = TrainGPR()
    print("\n")
    print("10-CV Test MAE:", mean_test_mae)
    print("10-CV Train MAE:", mean_train_mae)