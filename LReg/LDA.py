# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 22:31:45 2020

@author: AyeJay
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class LDA():
    _feat = None
    _targ = None
    _N = None
    _p_one = None
    _p_zero = None
    _u_one = None
    _u_zero = None
    _E_one = None
    _E_zero = None
    _E = None
    def __init__(self,features,targets):
        self._feat = features
        self._targ = targets
        self._N = targets.size #Expecting 1d Array of labels
        self._n_one = targets.sum()
        self._n_zero = (self._N-self._targ.sum())
        self._p_one = self._n_one/self._N
        self._p_zero = self._n_zero/self._N
        self._u_zero = 0
        self._u_one = 0
        for x in range(self._N): ##INDICATOR
            if self._targ[x] == 1:
                self._u_one += self._feat[x]
            else:
                self._u_zero += self._feat[x]
        self._u_zero = self._u_zero/self._n_zero #Expecting 1/0 labels passed on
        self._u_one = self._u_one/self._n_one
        self._E_one =0
        self._E_zero =0
        for x in range(self._N):
            if self._targ[x] == 1:
                self._E_one += np.outer((self._feat[x] - self._u_one),(self._feat[x] - self._u_one))/(self._N -2)
            else:
                self._E_zero += np.outer((self._feat[x] - self._u_zero),(self._feat[x] - self._u_zero))/(self._N -2)
        self._E = self._E_one + self._E_zero

    def predict(self,features):
        predResult = []
        a = np.log(self._p_one/self._p_zero)
        b = (np.dot(np.dot(self._u_one.T,np.linalg.inv(np.array(self._E,dtype=np.float32))),self._u_one)/2) # Bit weird to explain why I didnt power by -1
        c = (np.dot(np.dot(self._u_zero.T,np.linalg.inv(np.array(self._E,dtype=np.float32))),self._u_zero)/2)
        for x in range(features.shape[0]):
            d = np.dot(np.dot(features[x].T,np.linalg.inv(np.array(self._E,dtype=np.float32))),(self._u_one-self._u_zero))
#            d = ((features[x].T*np.power(self._E,-1)*(self._u_one-self._u_zero)))
            res = a-b+c+d
            if res > 0:
                predResult.append(1)
            else:
                predResult.append(0)
            
        return np.array(predResult)
    
    def Accu_eval(self,pL,tL):
        n=tL.size
        return np.mean(pL == tL)
    
def main():
    ########## LR FOR PERK
#    df2 = np.array(pd.read_csv('data/parkinsons.data', sep=',',header=0))
#    dff2 =np.delete(df2,0,1)
#    dff2 = np.delete(dff2,16,1)
#    dff3 = df2[:,17:18]
#    dff3 = dff3.flatten()
#    
#    x_tr, x_t, y_tr, y_t = train_test_split(dff2, dff3, test_size=0.2)
#    
#    x = LDA(x_tr,y_tr)
#    
#    l = x.predict(x_t)
#    print(x.Accu_eval(l,y_t))

##### LR for sonar BOTTOM    
    
    df = np.array(pd.read_csv('data/sonar.all-data', sep=',',header=0))
    print(df[:,-1:])
    target = df[:,60:]
    target = np.where(target!='R',0,target)
    target = np.where(target=='R',1,target)
    target = target.flatten()
    
    x_tr, x_t, y_tr, y_t = train_test_split(df[:,:60], target, test_size=0.2)
#
    x = LDA(x_tr,y_tr)
    
    l = x.predict(x_t)
    print(x.Accu_eval(l,y_t))
    
#    x.fit(x_tr,y_tr,0.3,1000)
#    
#    print(x.Accu_eval(x.predict(x_t),y_t))
#    
   
    #print(x.printW())
    
if __name__== "__main__":
    main()