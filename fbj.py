import pandas as pd
import numpy as np
import scipy.stats as stats
import sys
sys.path.append("/home/markchang/libsvm-3.18/python/")
from svmutil import *

def read_data(filename):
    data = pd.read_table(filename, sep=',', warn_bad_lines=True, error_bad_lines=True)
    data = np.asarray(data.values, dtype = float)
    col_mean = stats.nanmedian(data,axis=0)
    inds = np.where(np.isnan(data))
    data[inds]=np.take(col_mean,inds[1])
    #data=[np.concatenate((np.array([data[:,1]]).T,data[:,6:]),axis=1)]
    return data 

data = read_data('COLING.csv') 
X_train = data[:,6: ]
Y_train = data[:,1:6]


#print X_train
#print Y_train
svm_input_x = [  { i+1: xr[i] for i in range(xr.shape[0]) if not np.isnan(xr[i]) } for xr in X_train ]
svm_input_y = [ y for y in Y_train[:,0]]
prob  = svm_problem(svm_input_y, svm_input_x)
param = svm_parameter('-t 0 -c 4 -b 1')
#m = svm_train(prob, param)
