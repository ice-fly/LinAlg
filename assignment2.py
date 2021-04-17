# Imports
import scipy as sp
import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl

### New solve

M = int(input("Enter size M: "))
N = int(input("Enter size N: "))
A = np.random.rand(N,N)
B = np.random.rand(M,N)
print("A:")
print(A)
print(f" and it's shape is {A.shape}")
print("B:") 
print(B)
print(f" and it's shape is {B.shape}")
X_one = npl.solve(A,B)
print("Numpy Solve")
print(X_one)
print(f" and it's shape is {X_one.shape}")

### Old solve
def row_echelon(MatrixA):
    """ Return Row Echelon Form of matrix MatrixA """

    # if matrix MatrixA has no columns or rows,
    # it is already in REF, so we return itself
    r, c = MatrixA.shape
    if r == 0 or c == 0:
        return MatrixA

    # we search for non-zero element in the first column
    for i in range(len(MatrixA)):
        if MatrixA[i,0] != 0:
            break
    else:
        # if all elements in the first column is zero,
        # we perform REF on matrix from second column
        MatrixB = row_echelon(MatrixA[:,1:])
        # and then add the first zero-column back
        return np.hstack([MatrixA[:,:1], MatrixB])

    # if non-zero element happens not in the first row,
    # we switch rows
    if i > 0:
        ith_row = MatrixA[i].copy()
        MatrixA[i] = MatrixA[0]
        MatrixA[0] = ith_row

    # we divide first row by first element in it
    MatrixA[0] = MatrixA[0] / MatrixA[0,0]
    # we subtract all subsequent rows with first row (it has 1 now as first element)
    # multiplied by the corresponding element in the first column
    MatrixA[1:] -= MatrixA[0] * MatrixA[1:,0:1]

    # we perform REF on matrix from second row, from second column
    MatrixB = row_echelon(MatrixA[1:,1:])

    # we add first row and first (zero) column, and return
    return np.vstack([MatrixA[:1], np.hstack([MatrixA[1:,:1], MatrixB]) ])

print("Row Echelon of A:")
RE = row_echelon(A)
print(RE)
print(f" and it's shape is {RE.shape}")

X_two = npl.solve(RE,B)
print("Old Solve")
print(X_two)
print(f" and it's shape is {X_two.shape}")

### Difference

print("Error of Solves")
Error = (X_one-X_two)
print(Error)
print(f" and it's shape is {Error.shape}")

print(f"With a maximum error of {np.amax(Error)}, the computational error is unreasonable, and I should have just used Numpy from the start. :)")