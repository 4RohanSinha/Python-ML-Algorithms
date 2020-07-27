import numpy as np

class LinearRegression():
    def __init__(self, X, y):
        self.__X = X
        self.__y = y
        self.__X = np.insert(X, 0, [1]*len(self.__y), axis=1)
        self.__theta = np.zeros((self.__X.shape[1], 1))

    @property
    def X(self):
        return self.__X

    @property
    def y(self):
        return self.__y

    @property
    def theta(self):
        return self.__theta

    @staticmethod
    def load(filename, delimiter=",", skiprows=0):
        data = np.loadtxt(open(filename, "rb"), delimiter=delimiter, skiprows=skiprows)
        dataCols = data.shape[1]
        X = data[:, 0:dataCols-1]
        y = data[:, dataCols-1:]
        return LinearRegression(X, y)
        

    def train(self, alpha=0.01, num_iters=1500):
        self.__gradientDescent(alpha, num_iters)

    @staticmethod
    def hypothesis(X, theta):
        return np.dot(X, theta)

    @staticmethod
    def costFunction(X, y, theta):
        h = self.hypothesis(X, theta)
        m = len(y)
        J = (1/(2*m))*np.sum(np.square(h-y))
        return J

    #TODO: fix error with multivariate linear regression
    def __gradientDescent(self, alpha=0.01, num_iters=1500):
        m = len(self.__y)
        for i in range(num_iters):
            h = self.hypothesis(self.__X, self.__theta)
            errs = h - self.__y
            change = np.dot(self.__X.T, errs)
            self.__theta = self.__theta - alpha*(1/m)*change

    def predict(self, xpredict):
        finalX = np.insert(xpredict, 0, [1])
        return np.dot(finalX, self.__theta)


#load data: np.loadtxt(open("ex1data1.txt", "rb"), delimiter=",", skiprows=1)
#note on plotting data:
#setting ranges on axes: plt.xlim(start, end) or plt.ylim(start, end)
#plt.plot(x_values, y_values, 'rx') (change to ro for circles as points)
