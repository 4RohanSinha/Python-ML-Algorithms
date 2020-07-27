import numpy as np
import warnings
import gradientDescent as gd
import LinearRegression as LinReg

data = np.loadtxt(open("ex1data2.txt", "rb"), delimiter=",", skiprows=1)

X = (data[:, 0:2])
y = data[:, 2:]

reg_model = LinReg.LinearRegression(X, y)

reg_model.train()

print(reg_model.theta)

#print(reg_model.predict(np.array([[1650, 3]])))
#print(warnings.filters)
