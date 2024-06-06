import numpy as np


dt = 50e-3

A = np.array([[1, dt, dt**2/2],
              [0, 1, dt],
              [0, 0, 1] ])

C = np.array([[1,0,0]])
Q = np.diag([1e-4, 5e-2, 500]);
R = 1e-6

P = Q
for i in range(10000):
    P = A@P@A.transpose() + Q
    K = P@C.transpose()/(C@P@C.transpose()+R)
    P = (np.identity(3)-K@C)@P
    
print(f"{K}")
print(f"{P}")
    

