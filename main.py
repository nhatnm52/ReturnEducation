import pandas as pd 
import dataProcessing as dp
import linearRegression as lr
from sklearn.model_selection import train_test_split

#var for X
varOLS = ['educ','exper','expersq','black','KWW','south','smsa','married','reg661','reg662','reg663','reg664','reg665','reg666','reg667','reg668']


file = pd.read_excel('wage.xlsx')
data = pd.DataFrame(file)
data_train, data_test= train_test_split(data, test_size = 0.1, shuffle=False)

#data train
data_tr, X, Y = dp.dataClear(varOLS,data_train)

beta = lr.OLS(X, Y)
print('Beta:')
print(beta)

e, trainError, acc = lr.checkModel(beta, X, Y)
print('Train error: ', trainError)
print('Accurency train:', acc)

#Test
data_t, X_t, Y_t= lr.dataClear(varOLS,data_test)
e_t, trainError_t, acc_t = lr.checkModel(beta, X_t, Y_t)
print('Test error: ', trainError_t)
print('Accurency test:', acc_t)