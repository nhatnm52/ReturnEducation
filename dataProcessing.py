import numpy as np
from numpy import linalg as ln

#Remove outliers
def remove_outlier_std(df_in, col_name):
    mean = df_in[col_name].mean()
    std  = df_in[col_name].std() 
    step = 3

    fence_low  = mean - std*step
    fence_high = mean + std*step
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

#Process data
def dataClear(varlist,dataIn):
  lst = ['lwage']
  for it in varlist:
    lst.append(it)
  dataOut = dataIn[lst].dropna() 
  dataOut["married"][dataOut["married"] != 1] = 0
  dataOut = remove_outlier_std(dataOut, 'lwage')
  varData = dataOut[varlist] # ->data for X

  X = np.concatenate((np.ones(shape=(varData.shape[0],1)),np.array(varData)), axis=1)
  Y = np.array(dataOut['lwage']).reshape(dataOut.shape[0] ,1)

  return dataOut, X, Y
