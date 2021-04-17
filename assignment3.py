# Imports
import scipy as sp
import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl

N = int(input("Enter size N: "))
A = np.random.rand(N,N)
print("A:")
print(A)
print(f" and it's shape is {A.shape}")
try:
    print("Eigen values of A:")
    EV=npl.eig(A)[0]
    print(EV)
except Exception as e:
    print(e)