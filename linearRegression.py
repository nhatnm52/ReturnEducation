import numpy as np
from numpy import linalg as ln

#Ordinary least square
def OLS(X,Y):
  #beta = (X_t*X )^(-1) *X_t*Y
  beta = (np.linalg.inv(X.transpose() @ X)) @ X.transpose() @ Y

  return beta

#Check 
def checkModel(beta, X, Y):
  e = Y - X @ beta
  trainError = np.linalg.norm(e)**2 / e.shape[0]
  R_square = 1 - trainError / (np.linalg.norm(Y - np.full((Y.shape[0],1),Y.mean()))**2/Y.shape[0])

  return e, trainError, R_square