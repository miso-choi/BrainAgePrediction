from data_preprocessing import Preprocess
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')


### Hyperparameter Optimization of LightGBM (using GridSearchCV)
def get_best_params(model, params):
    grid_model = GridSearchCV(model, param_grid = params,
                             scoring = 'neg_mean_absolute_error', cv=10, return_train_score=True) 
    grid_model.fit(scaled_1abcd, y_target)
    mae = -1*grid_model.best_score_
    print('{0} 10-CV optimized MAE:{1}, optimized parameter:{2}'.format(model.__class__.__name__, np.round(mae,3), grid_model.best_params_))
    
    return grid_model.cv_results_, grid_model.best_estimator_


def get_result_df(lgbm_cv_results):    
    lgbm_result = pd.DataFrame(lgbm_cv_results)
    lgbm_result.drop(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time', 'params'],axis=1,inplace=True)
    lgbm_df = pd.concat([lgbm_result.iloc[:,0:7],lgbm_result.iloc[:,17],lgbm_result.iloc[:,30]],axis=1)
    lgbm_df.iloc[:,7:9] = -1*lgbm_df.iloc[:,7:9]
    
    return lgbm_df
    

if __name__ == "__main__":
    scaled_1abcd, y_target = Preprocess()

    lgbm = LGBMRegressor(random_state=43)
    params ={
    'num_leaves': [150,200],
    'min_data_in_leaf' : [50,70],
    'learning_rate' : [0.01,0.1,1],
    'boosting_type' : ['gbdt'],
    'objective' : ['regression'],
    'max_depth' : [15,20],
    'feature_fraction' : [0.3,0.7,1]
    }
    
    lgbm_cv_results, best_lgbm = get_best_params(lgbm, params)
    lgbm_df = get_result_df(lgbm_cv_results)
    print(lgbm_df)
