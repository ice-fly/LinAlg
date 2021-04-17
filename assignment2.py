# Imports
import scipy as sp
import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl


a = np.array([[1, 2], [3, 5]])
b = np.array([1, 2])
x = npl.solve(a, b)
x