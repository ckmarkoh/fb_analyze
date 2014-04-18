import pandas as pd
import numpy as np
import scipy.stats as stats
import sys
import os
import operator
sys.path.append(os.environ.get("LIBSVM_PATH"))
from svmutil import *

def read_data(filename):
    data = pd.read_table(filename, sep=',', warn_bad_lines=True, error_bad_lines=True)
    data = np.asarray(data.values, dtype = float)
    col_mean = stats.nanmedian(data,axis = 0)
    inds = np.where(np.isnan(data))
    data[inds] = np.take(col_mean,inds[1])
    #data=[np.concatenate((np.array([data[:,1]]).T,data[:,6:]),axis=1)]
    X_train = data[:,6: ]
    Y_train = data[:,1:6]
    svm_x = map(lambda xr: { i+1: xr[i] for i in range(xr.shape[0]) if not np.isnan(xr[i]) } , X_train )
    svm_y_ary = map( lambda i : [ y for y in Y_train[:,i]], range(Y_train.shape[1]) )
    return svm_x, svm_y_ary

def data_split(data,s1,s2):
    return data[:s1]+data[s2:] , data[s1:s2]

def run_svm_wrapper(xy_split, param):
    x_tr,x_val = xy_split[0]
    y_tr,y_val = xy_split[1]
    return run_svm(x_tr, y_tr, x_val, y_val, param)

def run_svm(x_tr, y_tr, x_val, y_val, param):
    m = svm_train(y_tr, x_tr, param+" -q" )
    _ , p_acc, _ = svm_predict(y_val, x_val, m," -q")
    return [p_acc[0],p_acc[1]]


def gen_c(idx=0,m_len=False):
    m_array=[0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30]
    if m_len:
        return len(m_array)
    else:
        return operator.itemgetter(idx)(m_array)

def svm_model_average(svm_x, svm_y, param, v_f, v_s):
    result = np.average( map( lambda j :  run_svm_wrapper( 
                           map( lambda d :  data_split( d ,j*v_s,(j+1)*v_s) , [svm_x, svm_y] ),param )
                            ,range(v_f)),axis=0) 
    print param,result
    return result

def svm_model_select(svm_x, svm_y_ary, param, yid):
    v_f=38
    v_s=1
    svm_y = svm_y_ary[yid]
    result = map( lambda i : ( gen_c(idx=i),  svm_model_average(svm_x, svm_y, param%(gen_c(idx=i)), v_f, v_s) )
                             ,range(gen_c(m_len=True)))
    return max(result, key=lambda x : x[1][0]) 

def main():
    svm_x, svm_y_ary = read_data('COLING.csv') 
  #  param = "-t 0 -c %s "
  #  result = map(lambda i: svm_model_select(svm_x, svm_y_ary, param, i), range(len(svm_y_ary)))
  #  for (c,d) in result:
  #      print "t:0, c:%s, acc:%s"%(c,d[0])

  #  print"----------------------------------"
    param = "-t 2 -c %s "
    result = map(lambda i: svm_model_select(svm_x, svm_y_ary, param, i), range(len(svm_y_ary)))
    for (c,d) in result:
        print "t:2, c:%s, acc:%s "%(c,d[0])

if __name__ == "__main__":
    main()
