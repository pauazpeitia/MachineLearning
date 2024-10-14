import argparse

import numpy as np
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

def create_features(x, order):
            features = x[:, np.newaxis]
            
            for i in range(2, order +1):
                features = np.hstack((features, (x ** i)[:, np.newaxis]))
            
            return features


x = np.array([1, 2, 3, 4, 5])  # Datos de entrada
order = 3  # Orden del polinomio que queremos generar
polynomial_features = create_features(x, order)
print(polynomial_features)
