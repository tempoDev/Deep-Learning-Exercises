from NEURONA_MP import MPNeuron 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import  train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargamos los datos
breast_cancer = load_breast_cancer()
X = breast_cancer.data
Y = breast_cancer.target

df = pd.DataFrame(X, columns=breast_cancer.feature_names)
print(df)

# Dividimos el conjunto de datos
X_train, X_test, Y_train, Y_test = train_test_split(df, Y, stratify=Y)

X_train_bin = X_train.apply( pd.cut, bins=2, labels=[1,0])
X_test_bin = X_test.apply( pd.cut, bins=2, labels=[1,0])

print(X_train_bin)

mp_neuron = MPNeuron()
mp_neuron.fit(X_train_bin.to_numpy(), Y_train)

print("TRAIN")
print(Y_train)

print(mp_neuron.threshold)

Y_pred = mp_neuron.predict(X_test_bin.to_numpy())

print(Y_pred)
accuracy = accuracy_score(Y_test, Y_pred)
print(accuracy)

print(confusion_matrix(Y_test, Y_pred)) 