# Imports
import scipy as sp
import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl

# Shamelessly stolen code from https://math.stackexchange.com/questions/3073083/how-to-reduce-matrix-into-row-echelon-form-in-numpy/3073117#3073117
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
print(f" and it's shape is {A.shape}")
B = np.array([[1,2],[3,4]])
print("B:") 
print(B)
print(f" and it's shape is {B.shape}")
X = np.array([5,6])
print("X:")
print(X)
print(f" and it's shape is {X.shape}")



### a
try:
    print("A x B:")
    print(np.cross(A,B))
    print(f"and it's shape is {np.dot(A,B).shape}")
except:
    print("is impossible due to mismatched dimensions")

try:
    print("B x A:")
    print((np.cross(B,A)))
    print(f"and it's shape is {np.dot(B,A).shape}")
except:
    print("is impossible due to mismatched dimensions")

### b
print("Row Echelon of A:")
print(row_echelon(A))
print(f" and it's shape is {row_echelon(A).shape}")

print("Row Echelon of B:")
print(row_echelon(B))
print(f" and it's shape is {row_echelon(B).shape}")

### c
A = np.array([[1,2],[3,4],[5,6]])
B = np.array([[1,2],[3,4]])

print("Upper of A:")
print(np.triu(A,1))
print(f" and it's shape is {np.triu(A,1)}")
print("Lower of A:")
print(np.tril(A,-1))
print(f" and it's shape is {np.tril(A,-1)}")
print("Diagonal of A:")
print(np.tril(np.triu(A)))
print(f" and it's shape is {np.tril(np.triu(A))}")

print("Upper of B:")
print(np.triu(B,1))
print(f" and it's shape is {np.triu(B,1)}")
print("Lower of B:")
print(np.tril(B,-1))
print(f" and it's shape is {np.tril(B,-1)}")
print("Diagonal of B:")
print(np.tril(np.triu(B)))
print(f" and it's shape is {np.tril(np.triu(B))}")

### d
print("Rank of A:")
print(npl.matrix_rank(A))

print("Rank of B:")
print(npl.matrix_rank(B))

### e
A = np.array([[1,2],[3,4],[5,6]])
B = np.array([[1,2],[3,4]])

print("Null of A:")
print(spl.null_space(A))

print("Null of B:")
print(spl.null_space(B))

### f
print("Transpose of A:")
print(np.transpose(A))

print("Transpose of B:")
print(np.transpose(B))

### g
A = np.array([[1,2],[3,4],[5,6]])
B = np.array([[1,2],[3,4]])

try:
    print("Inverse of A:")
    print(npl.inv(A))
except:
    print("is not possible")

try:
    print("Inverse of B:")
    print(npl.inv(B))
except:
    print("is not possible")

### h
print("Norm of A:")
print(npl.norm(A))

print("Norm of B:")
print(npl.norm(B))

### i
print("Kronecker product of A & B:")
print(np.kron(A,B))
print("Kronecker product of B & A:")
print(np.kron(B,A))

### j
try:
    print("Eigen values of A:")
    print(npl.eig(A)[0])
    print("Eigen vectors of A:")
    print(npl.eig(A)[1])
except:
    print("No eigen values for the the linear transformation over the real feild")

try:
    print("Eigen values of B:")
    print(npl.eig(B)[0])
    print("Eigen vectors of B:")
    print(npl.eig(B)[1])
except:
    print("No eigen values for the the linear transformation over the real feild")

