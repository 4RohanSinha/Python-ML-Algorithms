import numpy as np
import gradientDescent as gd

class LinearRegression():
    def __init__(self, X, y):
        self.__X = X
        self.__y = y
        self.__X = np.insert(X, 0, [1]*len(self.__y), axis=1)
        self.__theta = np.zeros((self.__X.shape[1], 1))

    def train(self, alpha=0.01, num_iters=1500):
        self.__theta = gd.gradientDescent(self.__X, self.__y, self.__theta, self.hypothesis, alpha, num_iters)

    @staticmethod
    def hypothesis(X, theta):
        return np.dot(X, theta)

    @staticmethod
    def costFunction(X, y, theta):
        h = self.hypothesis(X, theta)
        m = len(y)
        J = (1/(2*m))*np.sum(np.square(h-y))
        return J

    def predict(self, xpredict):
        finalX = np.insert(xpredict, 0, [1])
        return np.dot(finalX, self.__theta)

    @property
    def theta(self):
        return self.__theta

#load data: np.loadtxt(open("ex1data1.txt", "rb"), delimiter=",", skiprows=1)
#note on plotting data:
#setting ranges on axes: plt.xlim(start, end) or plt.ylim(start, end)
#plt.plot(x_values, y_values, 'rx') (change to ro for circles as points)
