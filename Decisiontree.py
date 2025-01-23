import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from data import Data as dt 
from collections import Counter

class Node : 
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def if_leaf_of_the_node(self):
        return self.value is not None

class DecisionTree: 
    def __init__(self, min_split=2, max_depth=100, n_feats=None):
        self.min_split = min_split
        self.max_depth = max_depth
        self.root = None
        self.n_feats = n_feats
        

    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)


    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # greedily select the best split according to information gain
        best_feat, best_thresh = self._get_best_criteria(X, y, feat_idxs)
        
        # grow the children that result from the split
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feat, best_thresh, left, right)
    

    def _most_common_label(self, y):
        counter = Counter(y)
        if len(counter) == 0:
            return None
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    def _get_best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_col= X[:, feat_idx]
            thresholds = np.unique(X_col)
            for threshold in thresholds:
                gain = self._information_gain(y, X_col, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh
    
    def _information_gain(self, y, X_col, split_thresh):
        parententropy = entropy(y)
        left_idxs, right_idxs = self._split(X_col, split_thresh)
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        n = len(y)
        nl, nr = len(left_idxs), len(right_idxs)
        el, er = entropy(y[left_idxs]), entropy(y[right_idxs])
        childentropy = (nl/n) * el + (nr/n) * er
        ig = parententropy - childentropy
        return ig
    
    def _split(self, X_col, split_thresh):
        if len(X_col) >2:
            left_idxs = np.argwhere(X_col <= split_thresh).flatten()
            right_idxs = np.argwhere(X_col > split_thresh).flatten()
        else:
            if X_col[0] == split_thresh:
                left_idxs = [0]
                right_idxs = [1]
            else:
                left_idxs = [0,1]
                right_idxs = []
          
        return left_idxs, right_idxs

    def explore_tree(self,x,node):
        if node.if_leaf_of_the_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self.explore_tree(x, node.left)
        return self.explore_tree(x, node.right)

    def predict(self, X):
        return [self.explore_tree(x, self.root) for x in X]


def entropy(y):
    y_int = np.array(y*100, dtype=np.int32)
    hist = np.bincount(y_int)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])
    
def is_accurate (y_true, y_pred):
    return np.mean(y_true == y_pred)


def main():
    data = pd.read_csv("./data/Realestate.csv", header=None)
    data = data.drop(data.columns[0], axis=1) ##drop the first column
    X = data.drop(7, axis=1).values
    y = data[7].values
    Data = dt(X, y)
    train, test = Data.splitData(0.8, 0.2)
    data = train.X
    MSE = []
    minimal_mse = 100000
    best_depth = 0

    for i in range(1,20):
        print(i/20, "%\n")
        clf = DecisionTree(max_depth=i)
        clf.fit(train.X,train.y) 
        y_pred = clf.predict(test.X)
        mse= mean_squared_error(test.y, y_pred)
        MSE.append(mse)
        if mse< minimal_mse:
            minimal_mse = mse
            best_depth = i

    
    print(f"Best depth: {best_depth} with MSE: {minimal_mse}")

    plt.plot(range(1,20), MSE)
    plt.title("MSE for Decision tree with Entropy according to depth")
    plt.xlabel("Depth")
    plt.ylabel("MSE")
    plt.show()



if __name__ == "__main__":
    main()

