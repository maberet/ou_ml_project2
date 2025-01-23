import numpy as np
import pandas as pd

import math
from collections import Counter
from sklearn.model_selection import train_test_split


class Data : 
    def __init__(self, X=None, y=None):
        self.data = None
        self.X, self.y = X, y

    def getData(self,file):
        self.data = pd.read_csv(file, header=None)
        self.data =self.data.drop(self.data.columns[0], axis=1) ##drop the first column
        self.X = self.data.drop(7, axis=1).values
        self.y = self.data[7].values
        
    def splitData(self,train_size, test_size):
        #here test_size is 25% of the 80% of the data, and it will be used for the validation set
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,train_size=train_size, test_size=test_size, random_state=11, shuffle=True)
        return Data(X_train, y_train), Data(X_test, y_test)

    def __str__(self):
        return f"Data: {self.data}"
    
   