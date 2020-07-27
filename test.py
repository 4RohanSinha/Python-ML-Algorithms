import numpy as np
import warnings
import LinearRegression as LinReg


#example of linear regression class
#replace "data1.txt" with data file
reg_model = LinReg.LinearRegression.load("data1.txt", ",", 1)

reg_model.train()

print(reg_model.theta)
