import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from data import Data as dt 

def distance(x1, x2):
    res = 0.0
    for i in range(len(x1)):
        res += (x1[i]-x2[i])*(x1[i]-x2[i])
    return res


def indexes (pt, train_data):
    indexes = []
    for i in range(len(train_data)):
        indexes.append(
            {"index": i, "distance": distance(train_data[i], pt)})
    return sorted(indexes, key=lambda d: d["distance"])


def knn_regression_uniform_weights(data_x, data_y, k,data):
    res = []
    for pt in data:
        distances = indexes(pt,data_x)
        sum = 0.0
        for i in range(k):
            sum += data_y[distances[i]["index"]]
        res.append(round(sum/k, 5))
    return res


def knn_weighted_regression(data_x, data_y, k,to_predict):
    res = []
    for pt in to_predict:
        distances = indexes(pt, data_x)
        if (distances[0]["distance"] > 0.00001):
            sum = 0.0
            weight_sum = 0.0
            for i in range(k):
                weight = 1.0/(distances[i]["distance"])
                sum += data_y[distances[i]["index"]]*weight
                weight_sum += weight
            res.append(round(sum/weight_sum, 5))
        else:
            res.append(data_y[distances[0]["index"]])
    return res


def KNN (): 
    data = pd.read_csv("./data/Realestate.csv", header=None)
    data = data.drop(data.columns[0], axis=1) ##drop the first column
    X = data.drop(7, axis=1).values
    y = data[7].values
    Data = dt(X, y)
    train, test = Data.splitData(0.8, 0.2)
    data = train.X

    x=[]
    y=[]

    best_neighbors = 0
    min_mse = 999.0
    Type = input("Enter the type of weights (uniform/weighted): ")
    for i in range(1, 71):
        if Type == "uniform":
            pred_y = knn_regression_uniform_weights(train.X, train.y, i, test.X)
        else:
            pred_y = knn_weighted_regression(train.X, train.y, i, test.X)
        mse = mean_squared_error(pred_y, test.y)
        x.append(i)
        y.append(mse)
        if i % 10 == 0:
            print("for ", i, " neighbours : ", mse)
        if mse < min_mse:
            best_neighbors = i
            min_mse = mse

        
    print("best results for ", best_neighbors, " neighbours : ", min_mse)

    plt.plot(x, y,marker = 'o',  alpha = 0.6)
    title = "MSE with distance weights" if Type == "weighted" else "MSE without distance weights"  
    plt.title(title)
    plt.xlabel("N° of neighbours")
    plt.ylabel("MSE")

    plt.show()


if __name__ == "__main__":
    KNN()