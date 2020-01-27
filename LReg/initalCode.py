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
        #target = np.where(target!=classOne,0,target)
        #target = np.where(target==classOne,1,target)
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
            #tData = np.insert(tData,0,1,axis=1)
            print(tData)
            print(corVec)
            counter = 0
            for e in tData:
                newW = []
                innerC = 0
                for w in self._weights:
                    z = w + lRate*((corVec[counter]-self.predict(e))*e[innerC])
                    newW.append(z)
                    innerC += 1
                self._weights = np.array(newW)
                counter +=1
        print(counter)
        print(self._weights)


    def predict(self,tSet):
        returnSet = []
        
        if tSet.ndim == 1: #Single training example
            power = 0
            counter = 0
            
            for x in tSet:
                
                power += x * self._weights[counter]

                counter +=1
            #print(1.0/(1.0+np.exp(-power)))
            returnSet.append(1/(1+np.exp(-power)))
        else: #Set of inputs
            for y in tSet:
                power = 0
                counter = 0
                for x in y:
                    
                    power += x * self._weights[counter]
                    counter +=1
                returnSet.append(1/(1+np.exp(-power)))
        if len(returnSet) > 1:
            return np.array(returnSet)
        else:
            return returnSet[0]
        
def main():
    df = np.array(pd.read_csv('data/test.data', sep=',',header=0))

    x = LRegression(df[:,1:],df[:,:1],'Male')

    print(x.fit(x.featt,x.tarr,0.2))
    

    

    
    
if __name__== "__main__":
    main()
