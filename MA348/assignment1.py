# %%
# Imports
import scipy as sp
import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl

# %% [markdown]
# # 1a)

# %%
x = 1
y = 1

print(x,y)

u = np.abs(np.sqrt(np.divide(x+np.sqrt(np.square(x)+np.square(y)),2)))
v = np.divide(y,np.multiply(2,u))
print("U isn't bad:\t",u)
print("V is worse:\t",v)

# %%
x = 100
y = 1

print(x,y)

u = np.abs(np.sqrt(np.divide(x+np.sqrt(np.square(x)+np.square(y)),2)))
v = np.divide(y,np.multiply(2,u))
print("U isn't bad:\t",u)
print("V is worse:\t",v)

# %%
x = -1
y = 1

print(x,y)

u = np.abs(np.sqrt(np.divide(x+np.sqrt(np.square(x)+np.square(y)),2)))
v = np.divide(y,np.multiply(2,u))
print("U is worse:\t",u)
print("V isn't bad:\t",v)

# %%
x = -100
y = 1

print(x,y)

u = np.abs(np.sqrt(np.divide(x+np.sqrt(np.square(x)+np.square(y)),2)))
v = np.divide(y,np.multiply(2,u))
print("U is worse:\t",u)
print("V isn't bad:\t",v)

# %% [markdown]
# # 2a&b)

# %%
x=1.1
u=np.power(x,3)-np.multiply(3,np.power(x,2))+np.multiply(3,x)
v=np.multiply(np.multiply((x-3),x)+3,x)
print("2a has correct answer:\t",u)
print("2b method is worse:\t",v)

# %% [markdown]
# # 2C)

# %%
print("relative error:\t",np.divide((v-u),u))

# %% [markdown]
# # 3
# 
# refrence:
# https://en.wikipedia.org/wiki/Trigonometric_functions#Power_series_expansion

# %%
x=np.arange(0, np.divide(np.pi,2), 1e-6) # get the range in sufficiently small steps to be equivalent to continuous
answer=np.cos(x) # Calculate answers
n=0 # Itterator
result=0 # Storage for resultant
#print(answer,result,np.isclose(answer,result,rtol=1e-06).all())
while not np.isclose(answer,result,rtol=1e-6).all():
    result+=np.multiply(np.divide(np.power(-1,n),np.math.factorial(np.multiply(2,n))),np.power(x,np.multiply(2,n)))
    n+=1
print("Result:\t",result," achieved after ",n,"terms")
