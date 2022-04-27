# %%
# Imports
import scipy as sp
import numpy as np
import random
import numpy.linalg as npl
import scipy.linalg as spl

# %% [markdown]
# # 1a)
# https://gist.github.com/cbellei/8ab3ab8551b8dfc8b081c518ccd9ada9

# %%
def TDMASolve(a, b, c, d):
    n = len(a)
    ac, bc, cc, dc = map(np.array, (a, b, c, d))
    xc = []
    for j in range(1, n):
        if(bc[j - 1] == 0):
            ier = 1
            return
        ac[j] = ac[j]/bc[j-1]
        bc[j] = bc[j] - ac[j]*cc[j-1]
    if(b[n-1] == 0):
        ier = 1
        return
    for j in range(1, n):
        dc[j] = dc[j] - ac[j]*dc[j-1]
    dc[n-1] = dc[n-1]/bc[n-1]
    for j in range(n-2, -1, -1):
        dc[j] = (dc[j] - cc[j]*dc[j+1])/bc[j]
    return dc

# %%
n=3

# %%
# Make Tridiagonal
a = np.array([2,3,0])
b = np.array([1,2,3])
c = np.array([0,1,2])
f = np.array([10,20,30])
A = np.vstack((a,b,c))
tdma=TDMASolve(a,b,c,f)
print(tdma)

# %% [markdown]
# # 1b)
# https://www.codesansar.com/numerical-methods/python-program-gauss-seidel-iteration-method.htm

# %%
def GaussSeidel(A, b, tolerance=1e-4):    
    x = np.zeros_like(b, dtype=np.single)
    x_old  = x.copy()
    
    #Loop over rows
    for i in range(A.shape[0]):
        x[i] = (b[i] - np.dot(A[i,:i], x[:i]) - np.dot(A[i,(i+1):], x_old[(i+1):])) / A[i ,i]
    #Iterate
    while npl.norm(x - x_old)/npl.norm(x) > tolerance:
        x_old  = x.copy()
        
        #Loop over rows
        for i in range(A.shape[0]):
            x[i] = (b[i] - np.dot(A[i,:i], x[:i]) - np.dot(A[i,(i+1):], x_old[(i+1):])) / A[i ,i]
            
    return x

# %%
gs=GaussSeidel(A,f)
print(gs)

# %% [markdown]
# # 1c)

# %%
print("relative error:\t",np.divide((gs-tdma),tdma))

print('Really big & really bad\n','Thomas algo is wayyyyy better')
