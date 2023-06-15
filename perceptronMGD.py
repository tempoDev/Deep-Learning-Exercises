from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Obtenemos la información
mnist = fetch_openml("mnist_784", as_frame=False)
#Lo pasamos a un dataframe de pandas
df = pd.DataFrame(mnist.data)


# Mostramos una selección de los datos
plt.figure(figsize=(20, 4))

for index, digit in zip( range(1,9), mnist.data[:8] ):
    plt.subplot(1,8, index)
    plt.imshow( np.reshape( digit, (28,28) ), cmap=plt.cm.gray )
    plt.title("Ejemplo " + str(index))
plt.show()

# Dividimos el conjunto de datos

X_train, X_test, Y_train, Y_test = train_test_split(mnist.data, mnist.target, test_size=0.15)

# Entrenamiento del algoritmo 
print("EMPEZANDO ENTRENAMIENTO")
print("...")
clf = MLPClassifier( hidden_layer_sizes=(100,), activation='logistic', solver="sgd" )
clf.fit(X_train, Y_train)
print("ENTRENAMIENTO FINALIZADO")

# Muestra de algunos parámetros del sistema
print("INFORMACIÓN DE LA RN")
print("CAPAS: " + str(clf.n_layers_))
print("OUTPUTS: " + str(clf.n_outputs_))
print("PARÁMETROS: " + str(clf.coefs_[0].shape))


# PREDICCIÓN
y_pred = clf.predict(X_test)
# F1 score de la predicción
print(f1_score(y_pred, Y_test, average="weighted"))

# Muestra de las imágenes mal clasificadas 

index = 0
index_errors = []
for label, predict in zip(y_pred, Y_test):
    if label != predict:
        index_errors.append(index)
    index += 1

plt.figure( figsize=(20, 4))

for index, img_index in zip( range(1, 9), index_errors[8:16] ):
    plt.subplot(1, 8, index)
    plt.imshow( np.reshape( X_test[img_index], (28, 28) ), cmap=plt.cm.gray)
    plt.title("Orig: " + Y_test[img_index] + " Pred: " + str(y_pred[img_index]))    
    
plt.show()