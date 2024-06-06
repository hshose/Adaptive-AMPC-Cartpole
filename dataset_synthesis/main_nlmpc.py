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

class Simulator:
    def __init__(self, timestep, Nsteps, f):
       
        x = ca.MX.sym('states', 4)
        u = ca.MX.sym('inputs', 1)

        self.dt = timestep
        self.N = Nsteps

        dae={'x':x, 'p':u, 'ode':f(x,u)}
        # ts = np.linspace(0, timestep, Nsteps)
        # opts = {}
        # opts["integrator"] = "cvodes"
        # opts["integrator_options"] = {"abstol":1e-10,"reltol":1e-10}
        # self.integrator = ControlSimulator("csim", f, tgrid, opts)
        opts = {}
        opts["reltol"] = 1e-9
        opts["abstol"] = 1e-9
        opts["fsens_err_con"] = True
        self.integrator = ca.integrator('integrator', 'cvodes', dae, 0, self.dt)

    def run(self, x0, u):
        x = np.zeros((4,self.N+1))
        x[:,0] = x0
        for i in range(self.N):
            sol = self.integrator(x0=x[:,i], p=u[:,i])
            x[:,i+1] = np.array(sol['xf']).flatten()
        return x

if __name__ == '__main__':
    # ---- setup controller     --------
    dt = 160e-3
    N = 25
       
    x0 = np.array([0, 0.99*np.pi, 0, 0])
    m1 = 0.0
    def f_true(x, u):
        return f(x,u,m1)
    
    sim = Simulator(dt, N, f_true)
    x1_np, u1_np, J = global_run(x0,N,dt)
    labels = ['mpc predict']
    U = [u1_np]
    X = []
    X.append(x1_np)
    print(f"{np.max(np.abs(x1_np[0,:]))}")
    
    plot_pendulum(np.linspace(0, dt*N, N+1), 9, U, X, labels, latexify=False)