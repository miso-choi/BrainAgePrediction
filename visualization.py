import numpy as np
from matplotlib import pyplot as plt

### Training and Validation MAE ( alpha & l1_ratio )
def plot_mae(elst_df):
    l1_ratio = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for i in l1_ratio:
        l1 = elst_df[elst_df['param_l1_ratio']==i]
        l1 = l1.reset_index(drop=True)

        train_mae = l1['mean_train_score'] 
        valid_mae = l1['mean_test_score']
        x=np.arange(0,7)
        labels=[0.0001,0.001,0.01,0.1,1,10,100]

        plt.figure(figsize=(6, 6)) 

        plt.plot(train_mae, label='Train MAE') 
        plt.plot(valid_mae, label='Validation MAE') 
        plt.xticks(ticks=x,labels=labels)

        plt.legend(loc='upper left',fontsize=10) 
        plt.ylabel('Mean Absolute Error',fontsize=12) 

        plt.ylim([4.5,15]) 
        plt.title('ElasticNet - Training and Validation MAE (l1_ratio={0})'.format(i),fontsize=12) 
        plt.xlabel('alpha',fontsize=12) 
        plt.show() 
