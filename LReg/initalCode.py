import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class LRegression():
    #Attributes
    _features = None
    _targets = None
    _weights = None
    featt = None
    tarr = None
    
    def __init__(self, feat,target):
        #TARGET IS EXPECTED TO BE changed to 1d array of 1 and zeros
        feat = np.insert(feat,0,1,axis=1) ##Adding x^0
        self._features = feat
        self._weights = np.zeros(self._features[0].size)
        self._targets = target
        self.tarr = self._targets.flatten()
        self.featt = self._features

    def fit(self, tData, corVec, lRate): 
        N = tData.shape[0]
        print(self.cost(corVec,self.sig(np.dot(tData,self._weights))))
        grad = np.dot((corVec-self.sig(np.dot(tData,self._weights))),tData) 
        grad *= lRate
        grad /= N
        self._weights = np.add(self._weights,grad)


    def predict(self,tSet):
        return self._classify(self.sig(np.array(np.dot(tSet, self._weights),dtype=np.float32)))
    
    def cost(self,trueV,predV):
        obs = trueV.shape[0]
        
        return ((-trueV*np.log(predV))-((1-trueV)*np.log(1-predV))).sum()/obs
    def sig(self,r):
        return 1.0/(1.0+np.exp(-(np.array(r,dtype=np.float32))))
    
    def printW(self):
        return self._weights
    def _classify(self,pred):
        l = []
        for x in pred:
            if x >= 0.5:
                l.append(1)
            else:
                l.append(0)
        return np.array(l)
        
    def Accu_eval(self,pL,tL):
        n=tL.size
        print(tL,pL)
        return np.mean(pL == tL)
    
def main():
    
#    df2 = np.array(pd.read_csv('data/parkinsons.data', sep=',',header=0))
#    print(df2)
    
    df = np.array(pd.read_csv('data/sonar.all-data', sep=',',header=0))
    print(df)
    
    target = df[:,60:]
    target = np.where(target!='R',0,target)
    target = np.where(target=='R',1,target)
    x = LRegression(df[:,:60],target)
    
    x_tr, x_t, y_tr, y_t = train_test_split(x.featt, x.tarr, test_size=0.2)

    for i in range(1000):
        x.fit(x_tr,y_tr,0.3)
    
    print(x.Accu_eval(x.predict(x_t),y_t))
    
   
    #print(x.printW())
    
    
    

    
    
if __name__== "__main__":
    main()
