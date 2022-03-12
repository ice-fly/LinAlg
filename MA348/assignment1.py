# Imports
import scipy as sp
import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl


x = 1
y = 1

print(x,y)

u = np.abs(np.sqrt(np.divide(x+np.sqrt(np.square(x)+np.square(y)),2)))
v = np.divide(y,np.multiply(2,u))
print("U isn't bad:",u)
print("V is worse:",v)


x = 100
y = 1

print(x,y)

u = np.abs(np.sqrt(np.divide(x+np.sqrt(np.square(x)+np.square(y)),2)))
v = np.divide(y,np.multiply(2,u))
print("U isn't bad:",u)
print("V is worse:",v)

x = -1
y = 1

print(x,y)

u = np.abs(np.sqrt(np.divide(x+np.sqrt(np.square(x)+np.square(y)),2)))
v = np.divide(y,np.multiply(2,u))
print("U is worse:",u)
print("V isn't bad:",v)

sigFigsError = np.multiply(0.5,10**(2-64))

x = -100
y = 1

print(x,y)

u = np.abs(np.sqrt(np.divide(x+np.sqrt(np.square(x)+np.square(y)),2)))
v = np.divide(y,np.multiply(2,u))
print("U is worse:",u)
print("V isn't bad:",v)

sigFigsError = np.multiply(0.5,10**(2-64))