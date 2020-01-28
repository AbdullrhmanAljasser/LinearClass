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
        #print("Previous weights")
        #print(self._weights)
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
            N = tData.shape[0]
            
            
            pred = self.predict(tData)
            print(self.cost(corVec,pred))
            grad = np.dot((corVec-pred),tData) 
            grad *= lRate
            grad /= N
            
            
            
            self._weights = np.add(self._weights,grad)
            #print(self._weights)
            #print(self._weights)
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
        #returnSet = []
        #if tSet.ndim == 1: #Single training example
        #    rSet = np.dot(tSet,self._weights)
        #else: #Set of inputs
        #    rSet = np.dot(tSet,self._weights)
        #return self.sig(rSet)
        return self.sig(np.array(np.dot(tSet, self._weights),dtype=np.float32))
    
    def cost(self,trueV,predV):
        obs = trueV.shape[0]
        
        return ((-trueV*np.log(predV))-((1-trueV)*np.log(1-predV))).sum()/obs
    def sig(self,r):
        return 1.0/(1.0+np.exp(-r))
    
    def printW(self):
        return self._weights
    def classify(self,pred):
        l = []
        for x in pred:
            if x >= 0.5:
                l.append(1)
            else:
                l.append(0)
        return np.array(l)
        
def main():
    df = np.array(pd.read_csv('data/sonar.all-data', sep=',',header=0))
    
    x = LRegression(df[:,:60],df[:,60:],'R')

    for i in range(1000):
        print(x.fit(x.featt,x.tarr,0.2))
    
    zz = x.predict(np.array([[1.0,0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032],[1,0.0453,0.0523,0.0843,0.0689,0.1183,0.2583,0.2156,0.3481,0.3337,0.2872,0.4918,0.6552,0.6919,0.7797,0.7464,0.9444,1.0000,0.8874,0.8024,0.7818,0.5212,0.4052,0.3957,0.3914,0.3250,0.3200,0.3271,0.2767,0.4423,0.2028,0.3788,0.2947,0.1984,0.2341,0.1306,0.4182,0.3835,0.1057,0.1840,0.1970,0.1674,0.0583,0.1401,0.1628,0.0621,0.0203,0.0530,0.0742,0.0409,0.0061,0.0125,0.0084,0.0089,0.0048,0.0094,0.0191,0.0140,0.0049,0.0052,0.0044]]))
    print(x.classify(zz))
    
    #print(x.printW())
    
    
    

    
    
if __name__== "__main__":
    main()
