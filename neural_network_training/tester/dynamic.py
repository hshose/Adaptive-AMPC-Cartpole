# from casadi import vertcat, sin, cos
import math


def get_param_rod():
    # M = 0.57 # mass of the cart [kg] -> now estimated
    # M = 0.875 # mass of the cart [kg] -> now estimated
    M = 0.506  # mass of the cart [kg] -> now estimated
    m = 0.23  # mass of the rod [kg] 0.227

    g = 9.81  # gravity constant [m/s^2]
    L = 0.6413  # length of the rod [m]
    J_pend = 7.88e-3
    return M, m, g, L, J_pend


def get_param_cart_motor():
    eta_g = 0.9  # planetar gearbox effeciency
    K_g = 3.71  # planetary gearbox ratio
    K_t = 7.68e-3  # motor current-torque constant [N m / A]
    R_m = 2.6  # motor armature resistance [Ohm]
    r_mp = 6.35e-3  # motor pinion radius [m]
    K_m = 7.68e-3  # motor bemf constant [V/(rad/s)]
    eta_m = 0.69  # motor efficiency
    B_eq = 4.3  # equivalent viscous damping coeficcient (cart) 4.3 [Nm s/rad]
    Bp = 2.4e-3  # equivalent viscous damping coeficcient (pendulum) [Nm s/rad]
    J_m = 3.9e-7  # motor rotor moment of interia [kg m^2]
    return eta_g, K_g, K_t, R_m, r_mp, K_m, eta_m, B_eq, Bp, J_m


def get_param_calc():
    eta_g, K_g, K_t, R_m, r_mp, K_m, eta_m, B_eq, B_p, J_m = get_param_cart_motor()
    A = (eta_g * K_g * K_t) / (R_m * r_mp)
    B = (-K_g * K_m) / r_mp
    C = eta_m
    J_mot_eq = eta_g * K_g ** 2 * J_m / r_mp ** 2
    return A, B, C, J_mot_eq, B_eq, B_p


M_cart_default, m_rod_default, _, L_rod_default, J_rod_default = get_param_rod()
A_default, B_default, C_default, J_mot_eq_default, B_eq_default, B_p_default = get_param_calc()
m_add_default = 0

m_add_opt = 0.0
M_cart_opt = 0.506
m_rod_opt = 0.23
L_rod_opt = 0.6393037858519218
J_rod_opt = 0.0
J_mot_eq_opt = 0.2153426947227305
AB_opt = -3.9610232930789304
AC_opt = 1.3002429170001302
B_eq_opt = 3.961023945944119
B_p_opt = 0.0002073799633807001


def f_rod_generic(
        x,
        u,
        m_add=0.02,
        M_cart=M_cart_default,
        m_rod=m_rod_default,
        L_rod=L_rod_default,
        J_rod=J_rod_default,
        J_mot_eq=J_mot_eq_default,
        AB=A_default * B_default,
        AC=A_default * C_default,
        B_eq=B_eq_default,
        Bp=B_p_default,
        cos=math.cos,
        sin=math.sin):
    g = 9.81
    # m_add = p

    L = L_rod

    m = m_add + m_rod
    l = (L / 2 * m_rod + L * m_add) / m

    # J = (m*l**2)/3
    J_rod = (m_rod * L ** 2) / 12
    # J_rod = (m_rod*L**2)/3
    J = J_rod + m_add * L ** 2

    M = M_cart + J_mot_eq

    h1 = M + m
    h2 = m * l
    h4 = m * l ** 2 + J
    h7 = m * l * g

    x1 = x[0]
    theta = x[1]
    v1 = x[2]
    dtheta = x[3]
    Vm = u[0]
    F = AB * v1 + AC * Vm

    # F = u[0]
    cos_theta = cos(theta)
    sin_theta = sin(theta)
    denominator = h2 ** 2 * cos_theta ** 2 - h1 * h4
    f_expl = [v1,
              dtheta,
              (h2 * h4 * dtheta ** 2 * sin_theta \
               - h2 * h7 * cos_theta * sin_theta \
               + h4 * (F - B_eq * v1) \
               - h2 * cos_theta * Bp * dtheta) / (-denominator),
              (h2 ** 2 * dtheta ** 2 * cos_theta * sin_theta \
               - h1 * h7 * sin_theta \
               + h2 * cos_theta * (F - B_eq * v1) \
               + h1 * Bp * dtheta) / denominator
              ]
    return f_expl


def f_rod_generic_m(x, u, parameter_pertubations):
    return f_rod_generic(x, u, m_add=0.02 + parameter_pertubations[0], M_cart=M_cart_opt + parameter_pertubations[2], m_rod=m_rod_opt, L_rod=L_rod_opt, J_rod=J_rod_opt,
                         J_mot_eq=J_mot_eq_opt, AB=AB_opt + parameter_pertubations[1],
                         AC=AC_opt + parameter_pertubations[3], B_eq=B_eq_opt, Bp=B_p_opt + parameter_pertubations[4])


if __name__ == "__main__":
    print(f"m_add:        default={m_add_default:.3f}  \t opt={m_add_opt}")
    print(f"M_cart:       default={M_cart_default:.3f}  \t opt={M_cart_opt}")
    print(f"m_rod:        default={m_rod_default:.3f}  \t opt={m_rod_opt}")
    print(f"L_rod:        default={L_rod_default:.3f}  \t opt={L_rod_opt}")
    print(f"J_rod:        default={J_rod_default:.3f}  \t opt={J_rod_opt}")
    print(f"J_mot_eq:     default={J_mot_eq_default:.3f}  \t opt={J_mot_eq_opt}")
    # print(f"A:            default={A_default:.3f}  \t opt={A_opt}")
    print(f"AB:            default={(A_default * B_default):.3f}  \t opt={AB_opt}")
    print(f"AC:            default={(A_default * C_default):.3f}  \t opt={AC_opt}")
    print(f"B_eq:         default={B_eq_default:.3f}  \t opt={B_eq_opt}")
    print(f"B_p:          default={B_p_default:.3f}  \t opt={B_p_opt}")
