from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df = pd.read_csv("final.csv")
p = pd.read_csv("experiencia.csv")
print(df.dtypes)
lin = linear_model.LinearRegression()
lin.fit(df[['anos_experiencia']], df.salario)
print(p.dtypes)
pred = lin.predict(p)
p['prediccion'] = pred
p.to_csv("resultado.txt", index= False)