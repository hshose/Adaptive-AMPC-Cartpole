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

import casadi as ca
from dynamicsgeneric import *


def f_rod(
        x, 
        u,
        m_add=0,
        M_cart=M_cart_default,
        m_rod=m_rod_default,
        L_rod=L_rod_default,
        J_rod = J_rod_default,
        J_mot_eq=J_mot_eq_default,
        AB=A_default*B_default,
        AC=A_default*C_default,
        B_eq=B_eq_default,
        Bp=B_p_default):
   
    f_expl_list = f_rod_generic( x=x, u=u, m_add=m_add, M_cart=M_cart, m_rod=m_rod, L_rod=L_rod, J_rod=J_rod, J_mot_eq=J_mot_eq, AB=AB, AC=AC, B_eq=B_eq, Bp=Bp, cos = ca.cos, sin = ca.sin)
    return ca.vertcat(f_expl_list[0], f_expl_list[1],f_expl_list[2], f_expl_list[3])

def E_kin(x, m_add=0, M_cart=M_cart_default, m_rod=m_rod_default, L_rod=L_rod_default, J_rod = J_rod_default, J_mot_eq=J_mot_eq_default, AB=A_default*B_default, AC=A_default*C_default, B_eq=B_eq_default, Bp=B_p_default):
    x1     = x[0]
    theta  = x[1]
    v1     = x[2]
    dtheta = x[3]
        

    g = 9.81
    L = L_rod
    m = m_add + m_rod
    l = (L/2*m_rod+L*m_add)/m
    m = m_add + m_rod
    l = (L/2*m_rod+L*m_add)/m
    
    J_rod = (m_rod*L**2)/12
    J = J_rod + m_add*L**2
    
    M = M_cart + J_mot_eq
   
    cos_theta = ca.cos(theta)
    sin_theta = ca.sin(theta)
    
    E_kin_cart  = 0.5*M*v1**2
    E_kin_pend = 0.5*m*( v1**2+2*v1*cos_theta*dtheta*l+l**2*dtheta**2) +0.5*J*dtheta**2
    
    return E_kin_cart + E_kin_pend

def E_pot(x, m_add=0, M_cart=M_cart_default, m_rod=m_rod_default, L_rod=L_rod_default, J_rod = J_rod_default, J_mot_eq=J_mot_eq_default, AB=A_default*B_default, AC=A_default*C_default, B_eq=B_eq_default, Bp=B_p_default):
    x1     = x[0]
    theta  = x[1]
    v1     = x[2]
    dtheta = x[3]
    
    g = 9.81
    L = L_rod
    
    m = m_add + m_rod
    l = (L/2*m_rod+L*m_add)/m
    
    return m*g*l*ca.cos(theta)

def f_rod_m(x,u,m):
    return f_rod(x,u,m_add=m,M_cart=M_cart_opt, m_rod=m_rod_opt, L_rod=L_rod_opt, J_rod = J_rod_opt, J_mot_eq=J_mot_eq_opt, AB=AB_opt, AC=AC_opt, B_eq=B_eq_opt, Bp=B_p_opt )

def E_kin_m(x,m):
    return E_kin(x,m_add=m, M_cart=M_cart_opt, m_rod=m_rod_opt, L_rod=L_rod_opt, J_rod = J_rod_opt, J_mot_eq=J_mot_eq_opt, AB=AB_opt, AC=AC_opt, B_eq=B_eq_opt, Bp=B_p_opt )

def E_pot_m(x,m):
    return E_pot(x,m_add=m, M_cart=M_cart_opt, m_rod=m_rod_opt, L_rod=L_rod_opt, J_rod = J_rod_opt, J_mot_eq=J_mot_eq_opt, AB=AB_opt, AC=AC_opt, B_eq=B_eq_opt, Bp=B_p_opt )


if __name__=="__main__":
    print(f"m_add:        default={m_add_default:.3f}  \t opt={m_add_opt}")
    print(f"M_cart:       default={M_cart_default:.3f}  \t opt={M_cart_opt}")
    print(f"m_rod:        default={m_rod_default:.3f}  \t opt={m_rod_opt}")
    print(f"L_rod:        default={L_rod_default:.3f}  \t opt={L_rod_opt}")
    print(f"J_rod:        default={J_rod_default:.3f}  \t opt={J_rod_opt}")
    print(f"J_mot_eq:     default={J_mot_eq_default:.3f}  \t opt={J_mot_eq_opt}")
    # print(f"A:            default={A_default:.3f}  \t opt={A_opt}")
    print(f"AB:            default={(A_default*B_default):.3f}  \t opt={AB_opt}")
    print(f"AC:            default={(A_default*C_default):.3f}  \t opt={AC_opt}")
    print(f"B_eq:         default={B_eq_default:.3f}  \t opt={B_eq_opt}")
    print(f"B_p:          default={B_p_default:.3f}  \t opt={B_p_opt}")