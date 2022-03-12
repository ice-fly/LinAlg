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

print("Orthonormal basis of A:")
OB=spl.orth(A)
print(OB)
print(f" and it's shape is {OB.shape}")