import numpy as np
from sklearn.metrics import accuracy_score
class MPNeuron:
    
    def __init__(self):
        self.threshold = None
    
    def Model(self, x):
        z = sum(x)
        return (z >= self.threshold)
    
    def predict (self, x):
        predicts = []
        for example in x:
            res = self.Model(example)
            predicts.append(res)
        return np.array(predicts)
    
    def fit(self, x, y):
        accuracy = {}
        for th in range(x.shape[1] + 1):
            self.threshold = th
            Y_pred = self.predict(x)
            accuracy[th] = accuracy_score(Y_pred, y)
        self.threshold = max(accuracy, key=accuracy.get)
    
mp_neuron = MPNeuron()
mp_neuron.threshold = 3

# False, True, False, True
prediccion = mp_neuron.predict([[1,0,0,1], [1,1,1,0], [0,0,0,0], [1,1,1,1]])
print(prediccion)
