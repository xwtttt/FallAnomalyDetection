import numpy as np 
import pandas as pd

def process(filename, label):
    df = pd.read_excel(filename)
    X = []
    y = []
    slices = []
    max_langth = 0
    for  index, row in df.iterrows():
        if pd.isnull(row).any():
            length = len(slices)
            if length >= 500:
                print(filename, index)
            if length > max_langth:
                max_langth = length
            X.append(slices)
            y.append(label)
            slices = []   
        else:
            slices.append(row)
    X.append(slices)
    y.append(label)
    return X, y, max_langth

filenames = ['act' + str(i) + '.xlsx' for i in range(6)]
labels = [0, 1, 2, 3, 4, 5]
Bigdata = np.full((444,500, 8), np.nan)
Biglabels = np.full((444,), np.nan)
bias = 0
for filename, label in zip(filenames, labels):
    X, y, _ = process(filename, label)
    assert len(X) == len(y)
    for i in range(len(X)):
        length = len(X[i])
        Bigdata[i+bias, 0:length,:] = np.array(X[i])
        Biglabels[i+bias] = y[i]
    bias += len(X)