import numpy as np
import pandas as pd
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler

def main():

    # load data with default settings
    X_train, X_val, y_train, y_val = utils.loadDataset(features=['Penicillin V Potassium 500 MG', 'Computed tomography of chest and abdomen', 
                                    'Plain chest X-ray (procedure)',  'Low Density Lipoprotein Cholesterol', 
                                    'Creatinine', 'AGE_DIAGNOSIS'], split_percent=0.8, split_state=42)

    # scale data since values vary across features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform (X_val)
    #print(X_train.shape, X_val.shape, y_val.shape, y_train.shape)
    #print(X_train[1:5])
    #print(y_train[1:5])
    
    """
    # for testing purposes once you've added your code
    # CAUTION: hyperparameters have not been optimized
    """
    ## change the hyperparameters
    log_model= logreg.LogisticRegression(num_feats=6, max_iter=100, tol=0.001, learning_rate=0.4, batch_size=50)
    #log_model = logreg.LogisticRegression(num_feats=6, max_iter=10, tol=0.01, learning_rate=0.00001, batch_size=12)
    log_model.train_model(X_train, y_train, X_val, y_val)
    log_model.plot_loss_history()
            


if __name__ == "__main__":
    main()
