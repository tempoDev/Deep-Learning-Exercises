from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split  
from sklearn.metrics import f1_score
from sklearn.linear_model import Perceptron 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

# Obtener dataset
#----------------------------------------------------------------
print("Obteniendo información...")
mnist = fetch_openml("mnist_784", as_frame=False)
print("Recibida.")
#----------------------------------------------------------------

# Datos
#----------------------------------------------------------------
print(mnist.data)

plt.figure(figsize=(20,4))
for index, digit in zip(range(1,9), mnist.data[:8]):
    plt.subplot(1,8, index)
    plt.imshow(np.reshape(digit, (28,28)), cmap=plt.cm.gray )
    plt.title("Ejemplo " + str(index))
plt.show()
df = pd.DataFrame(mnist.data,)
print(df)
#----------------------------------------------------------------

# División del conjunto de datos (10% de los datos)
#----------------------------------------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(mnist.data, mnist.target, test_size=0.1)
clf = Perceptron(max_iter=2000, random_state=40, n_jobs=1)
clf.fit(X_train, Y_train)

# Número de parámetros del modelo
print("PARAMETROS: ")
print(clf.coef_.shape)
# Parámetros del BIAS
print("BIAS: ")
print(clf.intercept_)
#----------------------------------------------------------------

# Predicción usando el conjunto de pruebas
#----------------------------------------------------------------
y_pred = clf.predict(X_test)
print( f1_score(Y_test, y_pred, average="weighted") )
#----------------------------------------------------------------

# Muestra de imágenes mal clasificadas
#----------------------------------------------------------------
index = 0
index_error = []
for label, predict in zip(Y_test, y_pred):
    if label != predict:
        index_error.append(index)
    index += 1
    
plt.figure(figsize=(20,4))
for i, img_index in zip(range(1,9), index_error[8:16]):
    plt.subplot(1, 8, i)
    plt.imshow( np.reshape(X_test[img_index], (28,28)), cmap=plt.cm.gray )
    plt.title("Orig: " + str(Y_test[img_index]) + "  Pred: " + str(y_pred[img_index]))
plt.show()
#----------------------------------------------------------------