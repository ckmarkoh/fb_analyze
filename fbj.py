import pandas as pd
import numpy as np
import scipy.stats as stats
import sys

def read_data(filename):
    data = pd.read_table(filename, sep=' ', warn_bad_lines=True, error_bad_lines=True)
    data = np.asarray(data.values, dtype = float)
    col_mean = stats.nanmedian(data,axis=0)
    inds = np.where(np.isnan(data))
    data[inds]=np.take(col_mean,inds[1])
    #data=[np.concatenate((np.array([data[:,1]]).T,data[:,6:]),axis=1)]
    return data 

data = read_data('COLING.csv') 
X_train = data[6:]
Y_train = data[:,1:6]


print X_train
print Y_train
sys.path.append("/home/markchang/Documents/libsvm-3.17/python")
