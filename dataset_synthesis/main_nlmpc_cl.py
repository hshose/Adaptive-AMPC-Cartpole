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
from dynamicscasadi import E_kin_m as E_kin
from dynamicscasadi import E_pot_m as E_pot
from nlmpc import *
from utils import *
from tqdm import tqdm
import casadi as ca

from utils import MaximumReinitializations

class Simulator:
    def __init__(self, timestep, f):
       
        x = ca.MX.sym('states', 4)
        u = ca.MX.sym('inputs', 1)

        self.dt = timestep
        self.N = 1

        dae={'x':x, 'p':u, 'ode':f(x,u)}
        # ts = np.linspace(0, timestep, Nsteps)
        # opts = {}
        # opts["integrator"] = "cvodes"
        # opts["integrator_options"] = {"abstol":1e-10,"reltol":1e-10}
        # self.integrator = ControlSimulator("csim", f, tgrid, opts)
        self.integrator = ca.integrator('integrator', 'cvodes', dae, 0, self.dt)

    def run(self, x0, u0):
        sol = self.integrator(x0=x0, p=u0)
        return np.array(sol['xf']).flatten()

if __name__ == '__main__':
    # ---- setup controller     --------
    # dt = 40e-3
    dt = 160e-3
    N  = 25
    m1 = 0.02
    m2 = 0.04
    c_wrong_mass    = Controller(N=N, dt=dt, m_add=m1)
    c_lin_corr_mass = Controller(N=N, dt=dt, m_add=m1)
    c_true_mass     = Controller(N=N, dt=dt, m_add=m2)
   
    
    # x_sample_max = np.array([0.35, np.pi, 1.2, 10])
    # x_sample_min = -x_sample_max
    # x0 =x_sample_min + np.random.rand(4) * ( x_sample_max - x_sample_min )
    
    x0 = np.array([0, 0.99*np.pi, 0, 0])
    # x0 = [0, 0.1, 0, 0]
    # x0 = [0.1, 0.3*pi, -0.5, 1]
    # x0 = [0.3, 0.5*pi, -0.5, -1]

    # M, J, H = c.run_taylor(x0)
    
    # m1 = 0.08

    def f_true(x, u):
        return f(x,u,m2)
    
    sim = Simulator(dt, f_true)
    
    N_sim = 50
    X_cl_wrong_mass = np.zeros((4,N_sim+1))
    U_cl_wrong_mass = np.zeros((1,N_sim))
    X_cl_lin_corr_mass = np.zeros((4,N_sim+1))
    U_cl_lin_corr_mass = np.zeros((1,N_sim))
    X_cl_true_mass = np.zeros((4,N_sim+1))
    U_cl_true_mass = np.zeros((1,N_sim))
    
    X_cl_wrong_mass[:,0] = x0
    X_cl_lin_corr_mass[:,0] = x0
    X_cl_true_mass[:,0] = x0
    
    failed_wrong_mass = False
    failed_lin_corr_mass = False
    failed_true_mass  = False
    
    for i in tqdm(range(N_sim)):
        if not failed_wrong_mass:
            try:
                print(f"\n\n\nMPC with wrong mass")
                # nominal with wrong mass
                X1, U1 = c_wrong_mass.run(X_cl_wrong_mass[:,i], sens=False)
                U_cl_wrong_mass[:,i] = np.array(U1[:,0])
                X_cl_wrong_mass[:,i+1] = np.array(sim.run(X_cl_wrong_mass[:,i], U_cl_wrong_mass[:,i]))
                # if i % 20== 0:
                    # plot_pendulum(np.linspace(0, dt*N, N+1), 9, [np.array(U1)], [np.array(X1)], ["wrong mass"], latexify=False)
            except MaximumReinitializations as e:
                print(e)
                print(f"Failed in wrong mass")
                failed_wrong_mass = True
        
        if not failed_lin_corr_mass:
            try:
                print(f"\n\n\nMPC with wrong mass and lin corr")
                # nominal with lin corr mass
                X2, U2, J2 = c_lin_corr_mass.run(X_cl_lin_corr_mass[:,i])
                U_cl_lin_corr_mass[:,i] = np.clip(np.array(U2[:,0]) + J2[0,0]*(m2-m1), -9, 9)
                X_cl_lin_corr_mass[:,i+1] = np.array(sim.run(X_cl_lin_corr_mass[:,i], U_cl_lin_corr_mass[:,i]))
            except MaximumReinitializations as e:
                print(e)
                print(f"Failed in lin corr")
                failed_lin_corr_mass = True
        
        if not failed_true_mass:
            try:
                print(f"\n\n\nMPC with correct mass")
                # true mass
                X3, U3 = c_true_mass.run(X_cl_true_mass[:,i], sens=False)
                U_cl_true_mass[:,i] = np.array(U3[:,0])
                X_cl_true_mass[:,i+1] = np.array(sim.run(X_cl_true_mass[:,i], U_cl_true_mass[:,i]))
            except MaximumReinitializations as e:
                print(e)
                print(f"Failed in true mass")
                failed_true_mass = True
        if i == 0:
                X2noncorr = np.zeros((4, N+1))
                X2noncorr[:,0] = x0
                
                U2corr = np.array(U2) + np.array(J2[0,:].reshape((1, N)))*(m2-m1)
                # print(f"U2s is same as U2corr? {np.all(np.abs(np.array(U2s)-U2corr)<=1e-4)}")
                X2corr = np.zeros((4, N+1))
                X2corr[:,0] = x0
                for i in range(N):
                    X2noncorr[:,i+1] = sim.run(X2noncorr[:,i], np.array(U2[:,i]))
                    X2corr[:,i+1] = sim.run(X2corr[:,i], U2corr[:,i])
                plot_pendulum(np.linspace(0, dt*N, N+1), 9, [np.array(U2),U2corr,np.array(U3)], [X2noncorr,X2corr, np.array(X3)], ["wrong mass", "lin corr", "true mass"], latexify=False)


    print(f"\n\n")
    print(f"Max cost on success was: {c_wrong_mass.max_cost_on_sucess}")
    print(f"Max cost on success was: {c_lin_corr_mass.max_cost_on_sucess}")
    print(f"Max cost on success was: {c_true_mass.max_cost_on_sucess}")
    print(f"\n\n")

    labels = ["wrong mass", "lin corr", "true mass"]
    U = [U_cl_wrong_mass, U_cl_lin_corr_mass, U_cl_true_mass]
    X = [X_cl_wrong_mass, X_cl_lin_corr_mass, X_cl_true_mass]
    plot_pendulum(np.linspace(0, dt*N_sim, N_sim+1), 9, U, X, labels, latexify=False)