# MIT License

# Copyright (c) 2024 Henrik Hose

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from time import time
from dynamicscasadi import f_rod_m as f
from nlmpc import *
from utils import *
import numpy as np
import casadi as ca
import math
import os

class Simulator:
    def __init__(self, timestep, Nsteps, f):
       
        x = ca.MX.sym('states', 4)
        u = ca.MX.sym('inputs', 1)

        self.dt = timestep
        self.N = Nsteps

        dae={'x':x, 'p':u, 'ode':f(x,u)}
        opts = {}
        opts["reltol"] = 1e-3
        opts["abstol"] = 1e-3
        opts["fsens_err_con"] = True
        self.integrator = ca.integrator('integrator', 'cvodes', dae, 0, self.dt)

    def run(self, x0, u):
        x = np.zeros((4,self.N+1))
        x[:,0] = x0
        for i in range(self.N):
            sol = self.integrator(x0=x[:,i], p=u[:,i])
            x[:,i+1] = np.array(sol['xf']).flatten()
        return x

def extract_data(x0_flat, X_flat, U_flat, J_flat, N):
    x0 = x0_flat
    U = U_flat.reshape((N, 1)).transpose()
    X = X_flat.reshape((N+1, 4)).transpose()
    J = J_flat.reshape((N, 5)).transpose()
    return x0, X, U, J

def eval_cost(X, U):
    x = X[0,:]
    theta = X[1,:]
    v = X[2,:]
    omega = X[3,:]

    m_add =0.02
    g = 9.81
    M_cart=0.506
    m_rod=0.23
    L_rod=0.6393037858519218
    J_mot_eq =0.2153426947227305
    AB=-3.9610232930789304
    AC=1.3002429170001302
    B_eq=3.961023945944119
    B_p=0.0002073799633807001
    
    L = L_rod
    J_rod = (m_rod*L**2)/12
    J = J_rod + m_add*L**2
    M = M_cart+J_mot_eq
    m = m_add+m_rod
    l = (L/2*m_rod+L*m_add)/m
    
    cost_horizon_scaling = 25/100/3
        
    cost = sum([
        0.5*M*v[t]**2                                     \
        +0.5*m*(                                          \
            v[t]**2                                       \
            +2*v[t]*math.cos(theta[t])*omega[t]*l         \
            +l**2*omega[t]**2)                            \
        +0.5*J*omega[t]**2                                \
        - m*g*l*math.cos(theta[t])                        \
        + 1e-2*U[0,t]**2                                  \
        + 1e-0*x[t]**2                                    \
            for t in range(len(U)) ])                     \
        + cost_horizon_scaling*1e2*theta[-1]**2
    return cost*3
    

if __name__ == '__main__':
    # ---- setup controller     --------
    dt = 160e-3
    N = 25
    
    m1 = 0.02
    
    def f_true(x, u):
        return f(x, u, m1)
    
    sim = Simulator(dt, N, f_true)
    
    datafilepath = os.path.join("data", "20231107-125758_dataset_3_small")
    
    
    x0_dataset = np.loadtxt( os.path.join(datafilepath, "x0.csv") , delimiter=',')[:1000]
    X_dataset  = np.loadtxt( os.path.join(datafilepath, "X.csv") , delimiter=',')[:1000]
    U_dataset  = np.loadtxt( os.path.join(datafilepath, "U.csv") , delimiter=',')[:1000]
    J_dataset  = np.loadtxt( os.path.join(datafilepath, "J.csv") , delimiter=',')[:1000]
    
    x0_min = np.min(x0_dataset, axis=0)
    x0_max = np.max(x0_dataset, axis=0)
    x0_spread = x0_max - x0_min
    
    N_samples = 100
    x0_dataset_normed = x0_dataset/x0_spread
    x0_dataset_remaining_normed = x0_dataset_normed[N_samples:]
    
    for i in range(N_samples):
        x0_normed_point = x0_dataset_normed[i]
        idx_near_remaining = np.linalg.norm(x0_dataset_remaining_normed-x0_normed_point, axis=1).argmin()
        idx_near = idx_near_remaining + N_samples
        
        x0_point, X_point, U_point, _ = extract_data(x0_dataset[i], X_dataset[i], U_dataset[i], J_dataset[i], N)
        x0_near,  X_near,  U_near,  _ = extract_data(x0_dataset[idx_near], X_dataset[idx_near], U_dataset[idx_near], J_dataset[idx_near], N)
        
        U_mean = np.mean(np.array([U_point, U_near]), axis=0)
        x0_mean = np.mean(np.array([x0_point, x0_near]), axis=0)
        X_mean = sim.run(x0_mean, U_mean)
        
        cost_point = eval_cost(X_point, U_point)
        cost_near =  eval_cost(X_near, U_near)
        cost_mean =  eval_cost(X_mean, U_mean)
        
        print(f"{cost_point=}")
        print(f"{cost_near=}")
        print(f"{cost_mean=}")
        
        labels = ["point", "near"]
        U = [U_point, U_near]
        X = [X_point, X_near]
        plot_pendulum(np.linspace(0, dt*N, N+1), 9, U, X, labels, latexify=False)