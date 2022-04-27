import math

from numpy import float32
track=0
for k in range(256): #range's stop is not included
    
    #print(''.join(['0','{:08b}'.format(k),'00000000000000000000000']))
    number=int(''.join(['0','{:08b}'.format(k),'00000000000000000000000']),2)/100
    small = float32(1)
    test=number+small
    while test != number:
        small/=2.0
        test=number+small
    if small>track:
        track=small
        print(small,"\t",number)