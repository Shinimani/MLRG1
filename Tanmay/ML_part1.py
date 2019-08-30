#!/usr/bin/python

import sys
import pandas as pd
import numpy as np
from sklearn import linear_model
import math

def cfun(x):
  return x * math.log(abs(x+0.02470001))

# define vectorized sigmoid
cfun_v = np.vectorize(cfun)


if sys.argv[1] == 'a':#part 1a
    trainfile = sys.argv[2]
    testfile = sys.argv[3]
    outputfile = sys.argv[4]
    weightfile = sys.argv[5]

    data = np.array(pd.read_csv(trainfile, header=None))#read training data
    test = np.array(pd.read_csv(testfile, header=None))#read test data
    n = data.shape[0]#number of training data
    m = data.shape[1] - 1#number of parameter excluding constant

    x = np.c_[np.ones(n),data[:,:(m)]]#add ones as parameter
    y = data[:,(m)]
    xtr = np.c_[np.ones(test.shape[0]),test]
    w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x),x)),np.transpose(x)),y)
    tres = np.dot(xtr,w)

    #loss function evaluation
    err = np.subtract(y,np.dot(x,w))
    sqerr = np.square(err)
    loss = (1/(2*n))*(np.sum(sqerr))
    norm_err = (np.sum(sqerr))/np.sum(np.square(y))

    np.savetxt(weightfile,w)
    np.savetxt(outputfile,tres)

elif sys.argv[1] == 'b':#part 1b
    trainfile = sys.argv[2]
    testfile = sys.argv[3]
    regularization = sys.argv[4]
    outputfile = sys.argv[5]
    weightfile = sys.argv[6]
    data = np.array(pd.read_csv(trainfile, header=None))#read training data
    test = np.array(pd.read_csv(testfile, header=None))#read test data
    reg = np.loadtxt(regularization)
    n = data.shape[0]#number of training data
    m = data.shape[1] - 1#number of parameter excluding constant

    x = np.c_[np.ones(n),data[:,:(m)]]#add ones as parameter
    #print(x)
    y = data[:,(m)]
    xtr = np.c_[np.ones(test.shape[0]),test]#add ones as parameter
    #find optimal lambda
    for lbda in reg:
        tot_err = 0
        for k in range(10):
            test_size = int(n/10)
            if k == 0:
                cv_test = x[0:(test_size),:]
                lb_test = y[0:test_size]
                cv_train = x[test_size:,:]
                lb_train = y[test_size:]
                w = np.dot(np.dot(np.linalg.inv(np.add(np.dot(np.transpose(cv_train),cv_train),(np.identity(cv_train.shape[1])*lbda))),np.transpose(cv_train)),lb_train)
                err = np.subtract(lb_test,np.dot(cv_test,w))
                sqerr = np.square(err)
                norm_err = (1/(2*n))*(np.sum(sqerr))
                tot_err += norm_err

            elif k == 9:
                cv_test = x[(n - test_size):,:]
                lb_test = y[(n - test_size):]
                cv_train = x[:(n - test_size),:]
                lb_train = y[:(n - test_size)]
                w = np.dot(np.dot(np.linalg.inv(np.add(np.dot(np.transpose(cv_train),cv_train),(np.identity(cv_train.shape[1])*lbda))),np.transpose(cv_train)),lb_train)
                err = np.subtract(lb_test,np.dot(cv_test,w))
                sqerr = np.square(err)
                norm_err = (1/(2*n))*(np.sum(sqerr))
                #norm_err = (np.sum(sqerr))/np.sum(np.square(lb_test))
                tot_err += norm_err
                #print('norm9 -> ',norm_err)

            else:
                cv_test = x[(k*test_size):((k+1)*test_size),:]
                lb_test = y[(k*test_size):((k+1)*test_size)]
                cv_train = np.concatenate((x[:(k*test_size),:],x[((k+1)*test_size):,:]))
                lb_train = np.concatenate((y[:(k*test_size)],y[((k+1)*test_size):]))
                w = np.dot(np.dot(np.linalg.inv(np.add(np.dot(np.transpose(cv_train),cv_train),(np.identity(cv_train.shape[1])*lbda))),np.transpose(cv_train)),lb_train)
                err = np.subtract(lb_test,np.dot(cv_test,w))
                sqerr = np.square(err)
                norm_err = (1/(2*n))*(np.sum(sqerr))
                #norm_err = (np.sum(sqerr))/np.sum(np.square(lb_test))
                tot_err += norm_err
                #print('norm',k,' -> ',norm_err)
        if lbda == reg[0]:
            min_lbda = lbda
            min_err = tot_err
        if min_err > tot_err:
            min_lbda = lbda
            min_err = tot_err
    #train on optimal lambda
    w = np.dot(np.dot(np.linalg.inv(np.add(np.dot(np.transpose(x),x),(np.identity(x.shape[1])*min_lbda))),np.transpose(x)),y)
    tres = np.dot(xtr,w)
    #print('w_shape ',w.shape,'tres shape ',tres.shape)

    np.savetxt(weightfile,w)
    np.savetxt(outputfile,tres)

elif sys.argv[1] == 'c':#part 1c
    trainfile = sys.argv[2]
    testfile = sys.argv[3]
    outputfile = sys.argv[4]

    data = np.array(pd.read_csv(trainfile, header=None))#read training data
    test = np.array(pd.read_csv(testfile, header=None))#read test data
    #print(data.shape,' ',test.shape)

    n = data.shape[0]#number of training data
    m = data.shape[1] - 1#number of parameter excluding constant

    x = data[:,:(m)]
    x2 = x ** 2
    normx2 = (x2 / x2.ptp(0))
    xptp2 = x2.ptp(0)
    con = 0.02470001
    logx = np.log(np.absolute(np.add(x,con*np.ones(x.shape))))
    normx = (x / x.ptp(0))
    xc = cfun_v(normx)
    xptp = x.ptp(0)
    print(x.shape == logx.shape)
    x = np.concatenate((x,x**2,logx,normx,normx2,xc),axis = 1)
    x = np.c_[np.ones(x.shape[0]),x]#add ones as parameter
    y = data[:,(m)]
    xtr = test
    xtr2 = xtr ** 2
    normxtr2 = (xtr2 / xptp2)
    logxtr = np.log(np.absolute(np.add(xtr,con*np.ones(xtr.shape))))
    normxtr = (xtr / xptp)
    xtrc = cfun_v(normxtr)
    xtr = np.concatenate((xtr,xtr**2,logxtr,normxtr,normxtr2,xtrc),axis = 1)
    xtr = np.c_[np.ones(xtr.shape[0]),xtr]#add ones as parameter

    alpha_list = np.array([0.3,0.1,0.03,0.02,0.015,0.01,0.007,0.003,0.001,0.0003,0.0001])
   # '''
    for alphaa in alpha_list:
        reg = linear_model.LassoLars(alpha = alphaa)
        reg.fit(x,y)
        if(alphaa == alpha_list[0]):
            min_error = reg.score(x,y)
            min_alpha = alphaa
        if(min_error < reg.score(x,y)):
            min_error = reg.score(x,y)
            min_alpha = alphaa
            print('change')
        print(alphaa,' score = ',reg.score(x,y))

    reg = linear_model.LassoLars(alpha = min_alpha)
    print(min_alpha)
    reg.fit(x,y)
    np.savetxt(outputfile,reg.predict(xtr))
    '''
    reg = linear_model.LassoLars(alpha=0.01)
    reg.fit(x,y)
    print('score = ',reg.score(x,y))
    np.savetxt(outputfile,reg.predict(xtr))
'''
