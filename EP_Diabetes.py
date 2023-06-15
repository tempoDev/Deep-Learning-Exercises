from NEURONA_MP import MPNeuron 
from sklearn.datasets import load_diabetes
from sklearn.model_selection import  train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

diabetes_data = load_diabetes()
X = diabetes_data.data
Y = diabetes_data.target

df = pd.DataFrame(X, columns=diabetes_data.feature_names)
print()

X_train, X_test, Y_train, Y_test = train_test_split(df, Y)


X_train_bin = X_train.apply( pd.cut, bins=2, labels=[1,0])
X_test_bin = X_test.apply( pd.cut, bins=2, labels=[1,0])


mp_neuron = MPNeuron()
mp_neuron.fit(X_train_bin.to_numpy(), Y_train)

print( "Threshold" + " " + str(mp_neuron.threshold))
print(Y_train)

Y_pred = mp_neuron.predict(X_test_bin.to_numpy())
#print(Y_pred)

accuracy = accuracy_score(Y_test, Y_pred)
print(accuracy)
print(confusion_matrix(Y_test, Y_pred)) 