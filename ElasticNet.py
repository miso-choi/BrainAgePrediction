from data_preprocessing import Preprocess
from visualization import plot_mae
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


### Hyperparameter Optimization of ElasticNet (using GridSearchCV)
def get_best_params(model, params):
    grid_model = GridSearchCV(model, param_grid = params,
                             scoring = 'neg_mean_absolute_error', cv=10, return_train_score=True) 
    grid_model.fit(scaled_1abcd, y_target)
    mae = -1*grid_model.best_score_
    print('{0} 10-CV optimized MAE:{1}, optimized parameter:{2}'.format(model.__class__.__name__, np.round(mae,3), grid_model.best_params_))
    
    return grid_model.cv_results_, grid_model.best_estimator_


def get_result_df(best_elasticnet_cv):
    elst_result = pd.DataFrame(best_elasticnet_cv)
    elst_result.drop(['param_fit_intercept','param_normalize','mean_fit_time','std_fit_time','mean_score_time','std_score_time','params'],axis=1, inplace=True)
    elst_df = pd.concat([elst_result.iloc[:,0:2],elst_result.iloc[:,12],elst_result.iloc[:,25]],axis=1)
    elst_df.iloc[:,2:4] = -1*elst_df.iloc[:,2:4]
    
    return elst_df



if __name__ == "__main__":
    scaled_1abcd, y_target = Preprocess()

    enet = ElasticNet()
    enet_params = {'alpha':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100], 'l1_ratio':[0, 0.2, 0.4, 0.6, 0.8, 1],
                   'fit_intercept':[True],'normalize':[False]}
    best_elasticnet_cv, best_elasticnet = get_best_params(enet, enet_params)
    elst_df = get_result_df(best_elasticnet_cv)
    plot_mae(elst_df)
    print(elst_df)
    
    