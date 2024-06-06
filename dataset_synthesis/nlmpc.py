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

import numpy as np
import copy
import pyomo.environ as pyo
import pyomo.dae as dae
import pyomo.contrib.sensitivity_toolbox.sens as pyosense

from utils import MaximumReinitializations

class FailedSensitivityException(Exception):
    pass

class FailedInitialOptimization(Exception):
    pass


class Controller:
    def __init__(self, N, dt, m_add=0.02):
        self.N = N
        self.dt = dt
        # m_add=0.0
        M_cart=0.506
        m_rod=0.23
        L_rod=0.6393037858519218
        J_mot_eq =0.2153426947227305
        AB=-3.9610232930789304
        AC=1.3002429170001302
        B_eq=3.961023945944119
        B_p=0.0002073799633807001

        model = pyo.ConcreteModel()
        
        model.m_add = pyo.Param(default=m_add, mutable=True)
        model.m_add_perturbed = pyo.Param(default=m_add+0.01, mutable=True)
        
        model.ABminusBeq = pyo.Param(default=AB-B_eq, mutable=True)
        model.ABminusBeq_perturbed = pyo.Param(default=AB-B_eq-1, mutable=True)
        
        model.M = pyo.Param(default=M_cart+J_mot_eq, mutable=True)
        model.M_perturbed = pyo.Param(default=M_cart+J_mot_eq+0.1, mutable=True)

        model.AC = pyo.Param(default=AC, mutable=True)
        model.AC_perturbed = pyo.Param(default=AC-0.3, mutable=True)
        
        model.B_p = pyo.Param(default=B_p, mutable=True)
        model.B_p_perturbed = pyo.Param(default=B_p*10, mutable=True)
        
        model.L = L_rod
        model.m = model.m_add + m_rod
        model.l = (model.L/2*m_rod+model.L*model.m_add)/model.m
        # J = (m*l**2)/3
        J_rod = (m_rod*model.L**2)/12
        model.J = J_rod + model.m_add*model.L**2
        # model.M = M_cart + J_mot_eq

        t_sim = np.linspace(0, dt*N, N+1)
        self.t_sim = t_sim
        model.g = 9.81


        model.t     = dae.ContinuousSet(initialize=t_sim)
        model.x     = pyo.Var(model.t)
        model.v     = pyo.Var(model.t)
        model.theta = pyo.Var(model.t)
        model.omega = pyo.Var(model.t)
        model.s     = pyo.Var(model.t, within=pyo.NonNegativeReals)
        # model.s     = pyo.Var(within=pyo.PositiveReals)
        model.u     = pyo.Var(model.t, bounds=(-9,9))

        # model.L    = pyo.Param(default=L)
        # model.l    = pyo.Param(default=l)
        # model.m    = pyo.Param(default=m)
        # model.J    = pyo.Param(default=J)
        # model.M    = pyo.Param(default=M)
        # model.AB   = pyo.Param(default=AB)
        # model.AC   = pyo.Param(default=AC)
        # model.B_eq = pyo.Param(default=B_eq)
        # model.B_p  = pyo.Param(default=B_p)


        model.xdot     = dae.DerivativeVar(model.x,     wrt=model.t)
        model.thetadot = dae.DerivativeVar(model.theta, wrt=model.t)
        model.vdot     = dae.DerivativeVar(model.v,     wrt=model.t)
        model.omegadot = dae.DerivativeVar(model.omega, wrt=model.t)

        model.xode = pyo.Constraint(model.t, rule=lambda model, t:
            model.xdot[t]==model.v[t])
        model.thetaode = pyo.Constraint(model.t, rule=lambda model, t:
            model.thetadot[t]==model.omega[t])
        
        def _vdot_diffeq(model, t):
            h1 = model.M + model.m
            h2 = model.m*model.l
            h4 = model.m*model.l**2+model.J
            h7 = model.m*model.l*model.g
            F = model.ABminusBeq*model.v[t]+model.AC*model.u[t]
            cos_theta = pyo.cos(model.theta[t])
            sin_theta = pyo.sin(model.theta[t])
            denominator = h2**2*cos_theta**2-h1*h4
            return model.vdot[t] == (h2*h4*model.omega[t]**2*sin_theta \
                        - h2*h7*cos_theta*sin_theta \
                        + h4*F \
                        - h2*cos_theta*model.B_p*model.omega[t] ) / (-denominator)
        
        def _omegadot_diffeq(model,t):
            h1 = model.M + model.m
            h2 = model.m*model.l
            h4 = model.m*model.l**2+model.J
            h7 = model.m*model.l*model.g
            F = model.ABminusBeq*model.v[t]+model.AC*model.u[t]
            cos_theta = pyo.cos(model.theta[t])
            sin_theta = pyo.sin(model.theta[t])
            denominator = h2**2*cos_theta**2-h1*h4
            return model.omegadot[t] == (h2**2*model.omega[t]**2*cos_theta*sin_theta \
                        - h1*h7*sin_theta \
                        + h2*cos_theta*F \
                        + h1*model.B_p*model.omega[t]) / denominator

        model.vode      = pyo.Constraint(model.t, rule=_vdot_diffeq)
        model.omegaode  = pyo.Constraint(model.t, rule=_omegadot_diffeq)
        
        model.x_lim_upper = pyo.Constraint(model.t, rule=lambda model, t:
            model.x[t] - 0.35 <= model.s[t])
        model.x_lim_lower = pyo.Constraint(model.t, rule=lambda model, t:
            - model.x[t] - 0.35 <= model.s[t])
        
        model.s_lim = pyo.Constraint(model.t, rule=lambda model, t:
            model.s[t] <= 0.01)
        
        
        # Terminal constraints
        model.x_lim_term_upper = pyo.Constraint(rule=lambda model:
            model.x[model.t[-1]] - 0.01 <= 0)
        model.x_lim_term_lower = pyo.Constraint(rule=lambda model:
            - model.x[model.t[-1]] - 0.01 <= 0)
        
        model.theta_lim_term_upper = pyo.Constraint(rule=lambda model:
            model.theta[model.t[-1]] - 0.01 <= 0)
        model.theta_lim_term_lower = pyo.Constraint(rule=lambda model:
            - model.theta[model.t[-1]] - 0.01 <= 0)
        
        model.v_lim_term_upper = pyo.Constraint(rule=lambda model:
            model.v[model.t[-1]] - 0.01 <= 0)
        model.v_lim_term_lower = pyo.Constraint(rule=lambda model:
            - model.v[model.t[-1]] - 0.01 <= 0)
        
        model.omega_lim_term_upper = pyo.Constraint(rule=lambda model:
            model.omega[model.t[-1]]  -  0.017 <= 0)
        model.omega_lim_term_lower = pyo.Constraint(rule=lambda model:
            - model.omega[model.t[-1]] - 0.017 <= 0)
        

        cost_mass_scaling = 1
        cost_horizon_scaling = self.N/100
        
        model.cost = sum([
            0.5*model.M*model.v[t]**2                               \
            +0.5*model.m*(                                          \
                model.v[t]**2                                       \
                +2*model.v[t]*pyo.cos(model.theta[t])*model.omega[t]*model.l   \
                +model.l**2*model.omega[t]**2)                      \
            +0.5*model.J*model.omega[t]**2                          \
            - model.m*model.g*model.l*pyo.cos(model.theta[t])       \
            + cost_mass_scaling*1e-2*model.u[t]**2                                    \
            + cost_mass_scaling*1e-0*model.x[t]**2                                    \
                for t in model.t ])                    \
            + cost_mass_scaling*cost_horizon_scaling*1e2*model.theta[model.t[-1]]**2 \

        model.cost_with_slack = model.cost + \
            sum([
            cost_mass_scaling*1e6*model.s[t]                                        \
            + cost_mass_scaling*1e6*model.s[t]**2                                   \
                for t in model.t ]
            )

        model.OBJ = pyo.Objective(expr=model.cost_with_slack, sense=pyo.minimize)
        self.model = model
        self.discretizer = pyo.TransformationFactory('dae.finite_difference')
        self.discretizer.apply_to(self.model, wrt=self.model.t, scheme='BACKWARD')

        self.opt = pyo.SolverFactory('ipopt', solver_io='nl')
        self.opt.options['linear_solver'] = 'ma57'
        self.opt.options['max_iter'] = 500
        self.opt.options['tol'] = 1e-12
        
        self.opt_sense = pyo.SolverFactory('ipopt_sens', solver_io='nl')
        self.opt_sense.options['run_sens'] = 'yes'
        self.opt_sense.options['linear_solver'] = 'ma57'
        self.opt_sense.options['tol'] = 1e-9

        self.do_random_init=True
        
        self.max_cost_on_sucess = 0
        
        
        self.param = [self.model.m_add, self.model.ABminusBeq, self.model.M, self.model.AC, self.model.B_p]
        self.perturbed_param = [self.model.m_add_perturbed, self.model.ABminusBeq_perturbed, self.model.M_perturbed, self.model.AC_perturbed, self.model.B_p_perturbed]


    
    def random_init(self):
        xmax = 0.35
        thetamax = np.pi
        vmax = 1
        omegamax = 1
        umax = 5
        
        xmin = -xmax
        thetamin = -thetamax
        vmin = -vmax
        omegamin = -omegamax
        umin = -umax
        
        for t in self.model.t:
            if t > 3.2:
                self.model.x[t].set_value(0, skip_validation=True)
                self.model.theta[t].set_value(0, skip_validation=True)
                self.model.v[t].set_value(0, skip_validation=True)
                self.model.omega[t].set_value(0, skip_validation=True)
                self.model.u[t].set_value(0, skip_validation=True)
            else:
                self.model.x[t].set_value( xmin + np.random.rand(1)*( xmax - xmin ), skip_validation=True)
                self.model.theta[t].set_value( thetamin + np.random.rand(1)*( thetamax - thetamin ), skip_validation=True)
                self.model.v[t].set_value( vmin + np.random.rand(1)*( vmax - vmin ), skip_validation=True)
                self.model.omega[t].set_value( omegamin + np.random.rand(1)*( omegamax - omegamin ), skip_validation=True)
                self.model.u[t].set_value(umin + np.random.randint(2,size=1)*(umax-umin), skip_validation=True)
            self.model.s[t].set_value(0, skip_validation=True)

    
    def get_solution(self, model):
        x = np.zeros((4, self.N+1))
        u = np.zeros((1, self.N+1))
        us = np.zeros((1, self.N+1))
        for i in range(len(self.t_sim)):
            t = self.t_sim[i]
            x[:,i]= np.array([
                pyo.value(model.x[t]),
                pyo.value(model.theta[t]),
                pyo.value(model.v[t]),
                pyo.value(model.omega[t])]
                ).flatten()
            u[:,i]=pyo.value(model.u[t])
            # if sens:
        return x, u
    
    def get_solution_sens(self, model):
        x = np.zeros((4, self.N+1))
        u = np.zeros((1, self.N+1))
        us = np.zeros((1, self.N+1))
        for i in range(len(self.t_sim)):
            t = self.t_sim[i]
            x[:,i]= np.array([
                pyo.value(model.x[t]),
                pyo.value(model.theta[t]),
                pyo.value(model.v[t]),
                pyo.value(model.omega[t])]
                ).flatten()
            u[:,i]=pyo.value(model.u[t])
            # if sens:
            us[:,i]=pyo.value(
                model.sens_sol_state_1[model.u[t]])
        return x, u, us
    
    def set_own_model_state(self, model):
        for t in self.model.t:
            self.model.x[t].set_value( pyo.value(model.x[t]), skip_validation=True)
            self.model.v[t].set_value( pyo.value(model.v[t]), skip_validation=True)
            self.model.theta[t].set_value( pyo.value(model.theta[t]), skip_validation=True)
            self.model.omega[t].set_value( pyo.value(model.omega[t]), skip_validation=True)
            self.model.s[t].set_value( pyo.value(model.s[t]), skip_validation=True)
            self.model.u[t].set_value( pyo.value(model.u[t]), skip_validation=True)
    
    
    def shift_solution(self):
        for i in range(3+1, len(self.model.t)+1):
            t_new = self.model.t[i-3]
            t_old = self.model.t[i]
            self.model.x[t_new].value     = pyo.value( self.model.x[t_old])
            self.model.theta[t_new].value = pyo.value( self.model.theta[t_old])
            self.model.v[t_new].value     = pyo.value( self.model.v[t_old])
            self.model.omega[t_new].value = pyo.value( self.model.omega[t_old])
            self.model.u[t_new].value     = pyo.value( self.model.u[t_old])
            
    def compute_dudp(self, u, us, p, p_perturbed):
        dp_value = p-p_perturbed
        du_dp = np.array([(u[0,i]-us[0,i])/dp_value for i in range(0,len(u[0,:]))]).reshape((1,self.N+1))
        return du_dp

    def compute_sensitivities(self):
        du_dp_arr = np.zeros((len(self.param), self.N+1))
        for i in range(len(self.param)):
            param = self.param[i]
            perturbed_param = self.perturbed_param[i]
            sens = pyosense.SensitivityInterface(self.model, clone_model=True)
            param_list = [param]
            perturb_list =[perturbed_param]
            sens.setup_sensitivity(param_list)
            m = sens.model_instance
            sens.perturb_parameters(perturb_list)
            results = self.opt_sense.solve(m, tee=False)
            try:
                _, u, us = self.get_solution_sens(m)
            except Exception as e:
                raise FailedSensitivityException(f"could not retrieve sensitivities: \nSolver {results=},\nError is {e} ")
            _, u_correct = self.get_solution(self.model)
            if not np.all(np.abs(u-u_correct)<=5e-3):
                raise FailedSensitivityException(f"u has changed after different sensitivity perturbation: {np.max(np.abs(u-u_correct))=}... wtf? ")
            du_dp_arr[i,:] = self.compute_dudp(u, us, param.value, perturbed_param.value)
        return du_dp_arr
    
    def run(self,x0, sens=True):
        N_rand_init=10
        for i in range(N_rand_init):
            print(f"NEXT TRY")
            try:
                if self.do_random_init:
                    print(f"reinitializing randomly!!!")
                    self.random_init()
                    self.opt.reset()
                    self.do_random_init=False
                self.model.x[0].fix(x0[0])
                self.model.theta[0].fix(x0[1])
                self.model.v[0].fix(x0[2])
                self.model.omega[0].fix(x0[3])
                
                results = self.opt.solve(self.model, tee=False)
                print(f"cost was {self.model.cost()}")
                
                if self.model.cost() >= 15:
                    raise FailedInitialOptimization(results)
                
                if results['Solver'][0]['Status'] == 'ok':
                    self.max_cost_on_sucess = max(self.max_cost_on_sucess, self.model.cost())
                    break
                else:
                    raise FailedInitialOptimization(results)
            except FailedInitialOptimization as e:
                print(f"cost was {self.model.cost()}")
                print(e)
                if i == N_rand_init-1:
                    print(f"FAILED IN {i+1}th attempt to reinit randomly")
                    raise MaximumReinitializations(e)
                self.do_random_init=True
        print(f"pass")
        # self.opt.solve(self.model, tee=True)
        x, u = self.get_solution(self.model)
        if sens:
            N_sens_tries = 10
            for i in range(N_sens_tries):
                try:
                    du_dp_arr = self.compute_sensitivities()
                except FailedSensitivityException as e:
                    self.opt_sense.reset()
                    print(f"reininitializing sense solver! restarting compute sense...")
                    if i == N_sens_tries-1:
                        raise MaximumReinitializations(e)
        self.shift_solution()
        
        if sens:
            return x, u[:,1:], du_dp_arr[:,1:]
        else:
            return x, u[:,1:]
        
    def run_random(self,x0, sens=True):
        N_rand_init=20
        res = []
        model_tries = []
        cost_tries = []
        for i in range(N_rand_init):
            try:
                print(f"reinitializing randomly!!!")
                self.random_init()
                self.opt.reset()
                self.model.x[0].fix(x0[0])
                self.model.theta[0].fix(x0[1])
                self.model.v[0].fix(x0[2])
                self.model.omega[0].fix(x0[3])
                
                results = self.opt.solve(self.model, tee=False)
                print(f"cost was {self.model.cost()}")
                if self.model.cost() >= 15:
                    raise FailedInitialOptimization(results)
                
                if results['Solver'][0]['Status'] == 'ok':
                    res.append(self.get_solution(self.model))
                    model_tries.append(self.model.clone())
                    cost_tries.append(self.model.cost_with_slack())
                else:
                    raise FailedInitialOptimization(results)
            except FailedInitialOptimization as e:
                pass

        if not cost_tries:
            raise MaximumReinitializations("FAILED IN {i+1}th attempt to reinit randomly")
        print(f"pass")
        # self.opt.solve(self.model, tee=True)
        print(f"\n choosing Min Cost {np.min(np.array(cost_tries))}")
        min_cost = np.min(np.array(cost_tries))
        idx_best = np.array(cost_tries).argmin()
        x, u = res[idx_best]
        best_model = model_tries[idx_best]
        self.set_own_model_state(best_model)
        if sens:
            N_sens_tries = 10
            for i in range(N_sens_tries):
                try:
                    du_dp_arr = self.compute_sensitivities()
                except FailedSensitivityException as e:
                    self.opt_sense.reset()
                    print(f"reininitializing sense solver! restarting compute sense...")
                    if i == N_sens_tries-1:
                        raise MaximumReinitializations(e)
        
        if sens:
            return x, u[:,1:], du_dp_arr[:,1:], min_cost
        else:
            return x, u[:,1:]
        
def global_run(x0, N, dt):
    res_try = []
    N_retries = 5
    N_fails_allowed = 2
    for i in range(N_retries):
        try:
            c = Controller(N=N, dt=dt)
            res_try.append(c.run_random(x0))
        except MaximumReinitializations as e:
            print(f"Error at initial condition {x0=}")
            print(e)
            
    if len(res_try) <= N_retries - N_fails_allowed:
        raise MaximumReinitializations(f'too many failed tries, {len(res_try)=}, skipping {x0=}')
    idx = np.array([res_try[i][3] for i in range(len(res_try))]).argmin()
    x1_np = res_try[idx][0]
    u1_np = res_try[idx][1]
    min_cost = res_try[idx][3]
    print(f"{min_cost=}")
    print(f"costs of tries: {[res_try[i][3] for i in range(len(res_try))]}")
    return res_try[idx][0], res_try[idx][1], res_try[idx][2]
        
        
