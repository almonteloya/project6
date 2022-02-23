
"""
Write your logreg unit tests here. Some examples of tests we will be looking for include:
* check that fit appropriately trains model & weights get updated
* check that predict is working

More details on potential tests below, these are not exhaustive
"""
import numpy as np
import pandas as pd
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler

def test_updates():
	# Check that your gradient is being calculated correctly
	
	# Check that your loss function is correct and that 
	# you have reasonable losses at the end of training
    
    # load data 
    X_train, X_val, y_train, y_val = utils.loadDataset(features=['Penicillin V Potassium 500 MG', 'Computed tomography of chest and abdomen', 
                                    'Plain chest X-ray (procedure)',  'Low Density Lipoprotein Cholesterol', 
                                    'Creatinine', 'AGE_DIAGNOSIS'], split_percent=0.8, split_state=42)

    # scale data since values vary across features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform (X_val)

    log_model= logreg.LogisticRegression(num_feats=6, max_iter=100, tol=0.001, learning_rate=0.4, batch_size=50)
    
    ## Test 1: Get the gradient before and after training  
    
    grad_before = log_model.calculate_gradient(X_train, y_train)
    log_model.train_model(X_train, y_train, X_val, y_val)
    grad_after = log_model.calculate_gradient(X_train, y_train)

    ## we check that the grad vectors are not the same
    assert(set(grad_before) != set(grad_after))
    
    # Test 2, check the loss is smaller once at the end
    log_model.train_model(X_train, y_train, X_val, y_val)
    assert(log_model.loss_history_train[-1]<log_model.loss_history_train[0])
    
    # Test 3: check that the loss after training is smaller than 1
    
    assert(log_model.loss_history_train[-1]<1)



def test_predict():
	# Check that self.W is being updated as expected
	# and produces reasonable estimates for NSCLC classification

	# Check accuracy of model after training

    X_train, X_val, y_train, y_val = utils.loadDataset(features=['Penicillin V Potassium 500 MG', 'Computed tomography of chest and abdomen', 
                                    'Plain chest X-ray (procedure)',  'Low Density Lipoprotein Cholesterol', 
                                    'Creatinine', 'AGE_DIAGNOSIS'], split_percent=0.8, split_state=42)

    # scale data since values vary across features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform (X_val)

    # Test 1, check that W is different before and after training
    log_model= logreg.LogisticRegression(num_feats=6, max_iter=100, tol=0.001, learning_rate=0.4, batch_size=50)
    W_before = log_model.W
    log_model.train_model(X_train, y_train, X_val, y_val)
    W_after = log_model.W
    assert(set(W_before) != set(W_after))
    
    #Test 2: check that preditions are between 1 - 0
    a=log_model.make_prediction(X_train)
    assert(np.all((a<1) & (a>0)) == True)
    
    # Test 3: check accuracy
    
    log_model = logreg.LogisticRegression(num_feats=6, max_iter=100, tol=0.001, learning_rate=0.4, batch_size=50)
    # Get the predcitions before the modeltrain
    predicted_notrain = np.where(log_model.make_prediction(X_val)>.5,1,0)

    log_model.train_model(X_train, y_train, X_val, y_val)
    
    # Predictions after training our model
    predicted_train = np.where(log_model.make_prediction(X_val)>.5,1,0)
    
    ## Accuracy test
    
    N = len(y_val)
    accuracy_before = ((y_val == predicted_notrain).sum()/N)
    accuracy_after = ((y_val == predicted_train).sum()/N)
    
    # The accuracyshould be higher after training
    assert(accuracy_after > accuracy_before)



