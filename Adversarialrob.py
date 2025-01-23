import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from data import Data as dt 

from KNN import knn_regression_uniform_weights
from KNN import knn_weighted_regression
from Decisiontree import DecisionTree as dc

def noisedataset(X,y, noise_level):
    n_samples, n_features = X.shape
    noise = np.random.normal(0, noise_level, size=(n_samples, n_features))
    return X+noise, y


def main():
    data = pd.read_csv("./data/Realestate.csv", header=None)
    data = data.drop(data.columns[0], axis=1) ##drop the first column
    X = data.drop(7, axis=1).values
    y = data[7].values
    Data = dt(X, y)
    train_data, test_data = Data.splitData(0.8, 0.2)
    noises=np.linspace(0, 10, 100)

    mses_KNN_U= []
    mses_DT=[]
    mses_KNN_W=[]
    clf = dc(max_depth=17)
    clf.fit(train_data.X, train_data.y)
    for noise in noises:
        X_test_noise, y_test_noise = noisedataset(test_data.X,test_data.y , noise)

        y_pred_with_noise = knn_regression_uniform_weights(train_data.X,train_data.y, 4, X_test_noise)
        mses_KNN_U.append(mean_squared_error(y_test_noise, y_pred_with_noise))

        y_pred_with_noise = knn_weighted_regression(train_data.X,train_data.y, 54, X_test_noise)
        mses_KNN_W.append(mean_squared_error(y_test_noise, y_pred_with_noise))
        
        y_pred_with_noise = clf.predict(X_test_noise)
        mses_DT.append(mean_squared_error(y_test_noise, y_pred_with_noise))


    plt.plot(noises, mses_KNN_U, label="KNN", color="red")
    plt.plot(noises, mses_DT, label="Decision Tree", color="blue")
    plt.plot(noises, mses_KNN_W, label="KNN weighted", color="green")
    legend_elements = [
            Patch(facecolor="red", edgecolor="black", label="KNN"),
            Patch(facecolor="green", edgecolor="black", label="KNN weighted"),
            Patch(facecolor="blue", edgecolor="black", label="Decision Tree")
        ]
    
    plt.legend(handles=legend_elements ,loc="upper right")
    plt.xlabel("Noise level")
    plt.ylabel("MSE")
    plt.title("MSE for KNN with uniform weights according to noise level")

    plt.show()




    return 0

if __name__ == "__main__":
    main()