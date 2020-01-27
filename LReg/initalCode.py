import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LRegression():
    #Attributes
    _features = None
    _targets = None
    _weights = None
    featt = None
    tarr = None
    
    def __init__(self, feat,target,classOne):
        ##Identifying classone and classzero in the target vector
        target = np.where(target!=classOne,0,target)
        target = np.where(target==classOne,1,target)
        feat = np.insert(feat,0,1,axis=1) ##Adding x^0
        self._features = feat
        self._weights = np.zeros(self._features[0].size)
        self._targets = target
        self.tarr = self._targets
        self.featt = self._features

    def fit(self, tData, corVec, lRate): 
        if corVec.ndim == 2: ##This vector shouldnt be 2d
            corVec = corVec.flatten()
            if corVec.size == 1:
                corVec = corVec[0]
        print("Previous weights")
        print(self._weights)
        if tData.ndim == 1: #Single traning example
            #Assumption that training data provided does not include
            #x_0 values
            #tData = np.insert(tData,0,1)
            counter = 0
            newW = []
            for w in self._weights:
                w += (lRate*(corVec-self.predict(tData)))*self.predict(tData)*(1-self.predict(tData))*tData[counter]
                newW.append(w)
                counter += 1
            self._weights = np.array(newW)
        else: #Multiple traning example
            N = tData[0].shape
            
            pred = self.predict(tData)
            print(tData.shape)
            grad = np.dot(tData.T, np.subtract(pred,corVec))
            print(grad.shape)
            grad /= N
            #print(grad)
            #print(tData[1])
            grad *= lRate
            
            
            self._weights = np.subtract(self._weights,grad)
            print(self._weights)
            #tData = np.insert(tData,0,1,axis=1)
#            print(tData)
#            print(corVec)
#            counter = 0
#            for e in tData:
#                newW = []
#                innerC = 0
#                
#                for w in self._weights:
#                    eSum = ((corVec[counter]-self.predict(e))*e[innerC])
#                    z = w + lRate*((corVec[counter]-self.predict(e*w))*e[innerC])
#                    newW.append(z)
#                    innerC += 1
#                self._weights = np.array(newW)
#                counter +=1
#        print(counter)
#        print(self._weights)


    def predict(self,tSet):
        returnSet = []
        if tSet.ndim == 1: #Single training example
            rSet = np.dot(tSet,self._weights)
        else: #Set of inputs
            rSet = np.dot(tSet,self._weights)
        return self.sig(rSet)
    
    def sig(self,r):
        return 1.0/(1.0+np.exp(-r))
        
def main():
    df = np.array(pd.read_csv('data/sonar.all-data', sep=',',header=0))
    
    x = LRegression(df[:,:60],df[:,60:],'R')

    for i in range(5000):
        print(x.fit(x.featt,x.tarr,0.3))
    
    
    
    

    
    
if __name__== "__main__":
    main()
