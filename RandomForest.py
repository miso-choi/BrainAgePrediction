from data_preprocessing import Preprocess
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


### Hyperparameter Optimization of RandomForest (using GridSearchCV)
def get_best_params(model, params):
    grid_model = GridSearchCV(model, param_grid = params,
                             scoring = 'neg_mean_absolute_error', cv=10, return_train_score=True) 
    grid_model.fit(scaled_1abcd, y_target)
    mae = -1*grid_model.best_score_
    print('{0} 10-CV optimized MAE:{1}, optimized parameter:{2}'.format(model.__class__.__name__, np.round(mae,3), grid_model.best_params_))
    
    return grid_model.cv_results_, grid_model.best_estimator_


def get_result_df(best_rf_cv):
    rf_result = pd.DataFrame(best_rf_cv)
    rf_result.drop(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time', 'params'],axis=1,inplace=True)
    rf_df = pd.concat([rf_result.iloc[:,0:5],rf_result.iloc[:,15],rf_result.iloc[:,28]],axis=1)
    rf_df.iloc[:,5:7] = -1*rf_df.iloc[:,5:7]  
    
    return rf_df


if __name__ == "__main__":
    scaled_1abcd, y_target = Preprocess()

    rf_reg = RandomForestRegressor(random_state=128)
    params= {
        'max_depth' : [30,60],     
        'n_estimators' : [50,100],
        'min_samples_leaf' : [3,6],
        'min_samples_split' : [4],
        'max_features' : [400]
        }
    
    best_rf_cv, best_rf = get_best_params(rf_reg, params)
    rf_df = get_result_df(best_rf_cv)
    print(rf_df)
    