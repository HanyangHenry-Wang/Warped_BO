import GPy
import numpy as np

X = np.array([[0.3],[0.5],[0.75],[0.8]])
Y = np.array([[0.1],[0.01],[0.2],[0.1]])


kernel = GPy.kern.RBF(input_dim=1)
m = GPy.models.GPRegression(X,Y,kernel)
m.Gaussian_noise.variance.fix(10**(-4))
m.optimize()