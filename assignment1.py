# Imports
import scipy
import numpy as np
import numpy.linalg as npl

# Shamelessly stolen code from https://math.stackexchange.com/questions/3073083/how-to-reduce-matrix-into-row-echelon-form-in-numpy/3073117#3073117
def row_echelon(A):
    """ Return Row Echelon Form of matrix A """

    # if matrix A has no columns or rows,
    # it is already in REF, so we return itself
    r, c = A.shape
    if r == 0 or c == 0:
        return A

    # we search for non-zero element in the first column
    for i in range(len(A)):
        if A[i,0] != 0:
            break
    else:
        # if all elements in the first column is zero,
        # we perform REF on matrix from second column
        B = row_echelon(A[:,1:])
        # and then add the first zero-column back
        return np.hstack([A[:,:1], B])

    # if non-zero element happens not in the first row,
    # we switch rows
    if i > 0:
        ith_row = A[i].copy()
        A[i] = A[0]
        A[0] = ith_row

    # we divide first row by first element in it
    A[0] = A[0] / A[0,0]
    # we subtract all subsequent rows with first row (it has 1 now as first element)
    # multiplied by the corresponding element in the first column
    A[1:] -= A[0] * A[1:,0:1]

    # we perform REF on matrix from second row, from second column
    B = row_echelon(A[1:,1:])

    # we add first row and first (zero) column, and return
    return np.vstack([A[:1], np.hstack([A[1:,:1], B]) ])



## Sizing
M = 3
N = 2
print("M:")
print(M)
print("N:") 
print(N)
## Matricies
A = np.array([[1,2],[3,4],[5,6]])
print("A:")
print(A)
print("and it's shape is {}".format(A.shape))
B = np.array([[1,2],[3,4]])
print("B:") 
print(B)
print(" and it's shape is {}".format(B.shape))
X = np.array([5,6])
print("X:")
print(X)
print(" and it's shape is {}".format(X.shape))



### a
try:
    print("A x B:")
    print(np.dot(A,B))
    print("and it's shape is {}".format(np.dot(A,B).shape))
except:
    print("is impossible due to mismatched dimensions")
try:
    print("B x A:")
    print((np.dot(B,A)))
    print("and it's shape is {}".format(np.dot(B,A).shape))
except:
    print("is impossible due to mismatched dimensions")

### b
print("Row Echelon of A:")
print(row_echelon(A))
print("Row Echelon of B:")
print(row_echelon(B))

### c
