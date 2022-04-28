# %%
# Imports
import scipy as sp
import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl

# %% [markdown]
# W=int[Pdv]
# Work equals the integral of pressure with respect to volumne

# %%
data=[{'P':336,'V':0.5},{'P':294.4,'V':2},{'P':266.4,'V':3},{'P':260.8,'V':4},{'P':260.5,'V':6},{'P':249.6,'V':8},{'P':193.6,'V':10},{'P':165.6,'V':11}]
area=0 # init area
data

# %%
def trapz(p1,p2): # Trapazoidal rule
    return np.multiply(np.divide(p2['V'] - p1['V'],2),(p2['P']+p1['P']))

# %%
def simp_third(p1,p2,p3): # Simpsons 1/3 rule
    return np.multiply((p3['V']-p1['V']),np.divide(p1['P']+np.multiply(4,p2['P'])+p3['P'],6))

# %%
def simp_eigth(p1,p2,p3,p4): # Simpsons 3/8 rule
    return np.multiply((p4['V']-p1['V']),np.divide(p1['P']+np.multiply(3,(p2['P']+p3['P']))+p4['P'],8))

# %% [markdown]
# Work equals integral from 0.5 to 11 of PdV 

# %% [markdown]
# 0,1 are trap
# 1,2,3 simpsons 1/3
# 3,5,6,7 sipsons 3/8
# 7,8 trap

# %%
area+=trapz(data[0],data[1])
print(area)

# %% [markdown]
# 1,2,3 simpsons 1/3

# %%
area+=simp_third(data[1],data[2],data[3])
print(area)

# %% [markdown]
# 3,4,5,6 sipsons 3/8

# %%
area+=simp_eigth(data[3],data[4],data[5],data[6])
print(area)

# %% [markdown]
# 6,7 trap

# %%
area+=trapz(data[6],data[7])
print(area)

# %% [markdown]
# Summed all together

# %%
print("Total work:\t",area,"kJ")


