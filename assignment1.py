# Imports
import copy
import scipy as sp
import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl

## Sizing
M = int(input("Enter size M: "))
N = int(input("Enter size N: "))
print("M:")
print(M)
print("N:") 
print(N)
## Matricies
Alpha = np.random.rand(N,N)
Bravo = np.random.rand(M,N)
Xray = np.random.rand(N,1)
def defineMatricies(Alpha,Bravo,Xray):
    return copy.copy(Alpha),copy.copy(Bravo),copy.copy(Xray)
[A,B,X] = defineMatricies(Alpha,Bravo,Xray) # Call definition
print("A:")
print(A)
print(f" and it's shape is {A.shape}")
print("B:") 
print(B)
print(f" and it's shape is {B.shape}")
print("X:")
print(X)
print(f" and it's shape is {X.shape}")

input("")
# Problem 1
print("\nProblem 1\n")
### a
try:
    print("A x B:")
    cross = np.cross(A,B)
    print(cross)
    print(f"and it's shape is {cross.shape}")
except:
    print("is impossible due to mismatched dimensions")

try:
    print("B x A:")
    cross = np.cross(B,A)
    print(cross)
    print(f"and it's shape is {cross.shape}")
except:
    print("is impossible due to mismatched dimensions")
input("")

### b
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

print("Row Echelon of A:")
RE = row_echelon(A)
print(RE)
print(f" and it's shape is {RE.shape}")

print("Row Echelon of B:")
RE = row_echelon(B)
print(RE)
print(f" and it's shape is {RE.shape}")

### c
[A,B,X] = defineMatricies(Alpha,Bravo,Xray) # Call definition

print("Upper of A:")
UP = np.triu(A,1)
print(UP)
print(f" and it's shape is {UP.shape}")
print("Lower of A:")
LO = np.tril(A,-1)
print(LO)
print(f" and it's shape is {LO.shape}")
print("Diagonal of A:")
DI= np.tril(np.triu(A))
print(DI)
print(f" and it's shape is {DI.shape}")

print("Upper of B:")
UP = np.triu(B,1)
print(UP)
print(f" and it's shape is {UP.shape}")
print("Lower of B:")
LO = np.tril(B,-1)
print(LO)
print(f" and it's shape is {LO.shape}")
print("Diagonal of B:")
DI= np.tril(np.triu(B))
print(DI)
print(f" and it's shape is {DI.shape}")
input("")

### d
[A,B,X] = defineMatricies(Alpha,Bravo,Xray) # Call definition
print("Rank of A:")
print(npl.matrix_rank(A))

print("Rank of B:")
print(npl.matrix_rank(B))
input("")

### e
[A,B,X] = defineMatricies(Alpha,Bravo,Xray) # Call definition

print("Null of A:")
NS = spl.null_space(A)
print(NS)
print(f" and it's shape is {NS.shape}")

print("Null of B:")
NS = spl.null_space(B)
print(NS)
print(f" and it's shape is {NS.shape}")
input("")

### f
[A,B,X] = defineMatricies(Alpha,Bravo,Xray) # Call definition
print("Transpose of A:")
TP = np.transpose(A)
print(TP)
print(f" and it's shape is {TP.shape}")

print("Transpose of B:")
TP = np.transpose(B)
print(TP)
print(f" and it's shape is {TP.shape}")
input("")

### g
[A,B,X] = defineMatricies(Alpha,Bravo,Xray) # Call definition

try:
    print("Inverse of A:")
    IN=npl.inv(A)
    print(IN)
    print(f" and it's shape is {IN.shape}")
except:
    print("is not possible")

try:
    print("Inverse of B:")
    IN=npl.inv(B)
    print(IN)
    print(f" and it's shape is {IN.shape}")
except:
    print("is not possible")
input("")

### h
[A,B,X] = defineMatricies(Alpha,Bravo,Xray) # Call definition
print("Norm of A:")
NM=npl.norm(A)
print(NM)
print(f" and it's shape is {NM.shape}")

print("Norm of B:")
NM=npl.norm(B)
print(NM)
print(f" and it's shape is {NM.shape}")
input("")

### i
[A,B,X] = defineMatricies(Alpha,Bravo,Xray) # Call definition
print("Kronecker product of A & B:")
KP=np.kron(A,B)
print(KP)
print(f" and it's shape is {KP.shape}")

print("Kronecker product of B & A:")
KP=np.kron(B,A)
print(KP)
print(f" and it's shape is {KP.shape}")
input("")

### j
[A,B,X] = defineMatricies(Alpha,Bravo,Xray) # Call definition
try:
    print("Eigen values of A:")
    EV=npl.eig(A)[0]
    print(EV)
    print(f" and it's shape is {EV.shape}")
    print("Eigen vectors of A:")
    EV=npl.eig(A)[1]
    print(EV)
    print(f" and it's shape is {EV.shape}")
except:
    print("No eigen values for the the linear transformation over the real feild")

try:
    print("Eigen values of B:")
    EV=npl.eig(B)[0]
    print(EV)
    print(f" and it's shape is {EV.shape}")
    print("Eigen vectors of B:")
    EV=npl.eig(B)[1]
    print(EV)
    print(f" and it's shape is {EV.shape}")
except:
    print("No eigen values for the the linear transformation over the real feild")
input("")

### k
[A,B,X] = defineMatricies(Alpha,Bravo,Xray) # Call definition
try:
    print("Singular Value Decomposition of A:")
    SV=npl.svd(A)[1]
    print(SV)
    print(f" and it's shape is {SV.shape}")
except:
    print("SVD computation does not converge")

try:
    print("Singular Value Decomposition of B:")
    SV=npl.svd(B)[1]
    print(SV)
    print(f" and it's shape is {SV.shape}")
except:
    print("SVD computation does not converge")
input("")

### l
# Shamelessly stolen code from https://stackoverflow.com/questions/10871220/making-a-matrix-square-and-padding-it-with-desired-value-in-numpy
def squarify(M,val=0):
    (a,b)=M.shape
    if a>b:
        padding=((0,0),(0,a-b))
    else:
        padding=((0,b-a),(0,0))
    return np.pad(M,padding,mode='constant',constant_values=val)

[A,B,X] = defineMatricies(Alpha,Bravo,Xray) # Call definition

print("Determinant of A:")
print(npl.det(squarify(A)))

print("Determinant of B:")
print(npl.det(squarify(B)))
input("")

### m
print("V is equal to:")
V=np.dot(B,X)
print(V)
print(f" and it's shape is {V.shape}")
input("")


# Problem 2
print("\nProblem 2\n")

### a
print("Identity (NxN):")
ID=np.identity(N)
print(ID)
print(f" and it's shape is {ID.shape}")
input("")

### b
print("Zeros (NxN):")
ZR=np.zeros((N, N))
print(ZR)
print(f" and it's shape is {ZR.shape}")
input("")

### c
print("Block Diagonal of A & B:")
BD=spl.block_diag(A,B)
print(BD)
print(f" and it's shape is {BD.shape}")
input("")

# Problem 3
print("\nProblem 3\n")
### a
A = []
for i in range(0, N):
    B = []
    for j in range(0, N):
        if ((i+j+1)%N==0):
            B.append(1)
        else:
            B.append(0)
    A.append(B)
A = np.matrix(A)
print("Antidiagonal Matrix:")
print(A)
print(f" and it's shape is {A.shape}")
input("")

### b
A = []
for i in range(0, N):
    B = []
    C = np.random.randint(0,N)
    for j in range(0, N):
        if (j==i):
            B.append(1)
        else:
            B.append(0)
    A.append(B)
A = np.matrix(A)
for i in range(0, N):
    if bool(np.random.randint(0,2)):
        A[:, [0, i]] = A[:, [i, 0]]
print("Permutation Matrix:")
print(A)
print(f" and it's shape is {A.shape}")
input("")

### c
Theta = float(input("Enter Theta: "))
R = []
for i in range(0, 2):
    for j in range(0, 2):
        B =[]
        if ((i+j)%2==0):
            B.append(np.cos(Theta))
        else:
            B.append((1-(2*j))*np.sin(Theta))
        R.append(B)
R = np.matrix(R).reshape(2,2)
print("2D rotation Matrix:")
print(R)
print(f" and it's shape is {R.shape}")

Theta = float(input("Enter Theta: "))
R = []
for i in range(0, 3):
    for j in range(0, 3):
        B =[]
        if (i==2 and j==2):
            B.append(1)
        elif(j==2 or i==2):
            B.append(0)
        elif ((i+j)%2==0):
            B.append(np.cos(Theta))
        else:
            B.append((1-(2*j))*np.sin(Theta))
        R.append(B)
R = np.matrix(R).reshape(3,3)
print("3D rotation Matrix:")
print(R)
print(f" and it's shape is {R.shape}")
input("")