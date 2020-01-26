import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression():
    #Attributes
    _features = None
    _targets = None
    weights = None
    
    #*args is used to check if header exists in the dataset provided
    def __init__(self, feat,target):
        for x in feat:
            x.put(0,1)
        self._features = np.array(feat)
        new = []
        for x in target:
            if x == 'R':
                new.append([1])
            else:
                new.append([0])
        self._targets = np.array(new)
        
        
        #Creating weight vector
        self.weights = np.power((self._features.T.dot(self._features)),-1).dot((self._features.T.dot(self._targets)))
        
        
def main():
    df = np.array(pd.read_csv('data/sonar.all-data', sep=',',header=None))
    
    xy = LinearRegression(df[:,:60],df[:,-1:])
    
    x = np.arange(0,10)
    y = 0
    for i in range(0,60):
        y = y + (x**60-i)*xy.weights[60-(60-i)][0]
    plt.plot(x,y) 
    plt.show()
    
    
if __name__== "__main__":
    main()
