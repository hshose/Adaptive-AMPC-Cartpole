import yaml
import jax.numpy as jnp
import jax

from data.dataset import denormalize, normalize
from neural_networks.jax_models import AMPCNN, AMPCAUGNN

import pickle

from time import time
from tester.dynamic import f_rod_generic_m as f
from tester.utils import *

from data.dataset import AMPCDataset

from scipy.integrate import odeint

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
import matplotlib as mpl


import scipy.stats


def heatmap(parameters, values, results, title, ax):
    colors = [(0.8, 0.8, 1), (0, 0, 0.5)]  # Light blue to dark blue
    cmap_name = 'blue_custom'
    custom_cmap = mpl.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

    im = ax.imshow(results,
                   interpolation='nearest',
                   aspect=0.2,
                   cmap=custom_cmap,
                   vmin=0,
                   vmax=1)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(parameters)), labels=parameters)
    ax.set_yticks(np.arange(len(values)), labels=["%.2f" % v for v in values])

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(values)):
        for j in range(len(parameters)):
            print(results.shape)
            print(i)
            print(j)
            text = ax.text(j, i, results[i, j],
                        ha="center", va="center", color="w")

    ax.set_title(title)


def plot_heatmap(values, results, results2):
    fig, ((ax, ax2)) = plt.subplots(1, 2, figsize=(40, 20))
    # fig, ax = plt.subplots()

    # plt.style.use("ggplot")

    parameters = [r"$m_\mathrm{add}$", r"$C_1$", r"$M$", r"$C_2$",
                  r"$C_3$"]

    heatmap(parameters, values, results.transpose(), "", ax)
    heatmap(parameters, values, results2.transpose(), "no gradients", ax2)

    ax.set_title(r"\textbf{with gradients (this paper)}", fontweight='bold')
    ax2.tick_params(axis='y', labelleft=False)
    ax.set_ylabel("rel. parameter deviation")
    ax.set_xlabel("parameter")
    ax2.set_xlabel("parameter")

    # fig.subplots_adjust(bottom=-2)
    # fig.tight_layout()
    plt.show()


def mod_pendulum(x):
    return jnp.array([x[0], jnp.mod(x[1] + jnp.pi, 2*jnp.pi) - jnp.pi, x[2], x[3]])


def call_control(x, model, model_aug, normalization, parameter_pertubations, augment_input=True):
    # x = mod_pendulum(x)

    x = normalize(x, normalization["x"])
    u = model(x)

    if model_aug is not None and augment_input:
        grad = model_aug(x)

        grad = denormalize(grad, normalization["gradient"])
        # print(grad)
        # print(jnp.array(parameter_pertubations) @ grad.T)
        # print(denormalize(u, normalization["u"]))
    u = denormalize(u, normalization["u"]) + (jnp.array(parameter_pertubations) @ grad.T if augment_input and model_aug is not None else 0)
    return jnp.clip(u, a_min=-9, a_max=9)


def simulate_plant(x0, sim, model, normalization, params, model_aug=None):
    N_sim = 60

    X = np.zeros((len(x0), 4, N_sim + 1))
    U = np.zeros((len(x0), 1, N_sim))

    X[:, :, 0] = x0

    def call_control_wrapper(x_):
        x_ = mod_pendulum(x_)
        return call_control(x_, model, model_aug, normalization, params["add_weight"], augment_input=True)

    call_control_v = jax.vmap(call_control_wrapper)

    for i in range(N_sim):
        U[:, :, i] = call_control_v(X[:, :, i])
        for j in range(len(x0)):
            X[j, :, i+1] = np.array(sim.run(X[j, :, i], U[j, :, i]))

    return X


def check_constraints(X):
    def check_constraints_(x_):
        x_ = mod_pendulum(x_)
        const = jnp.array([0.4, 100, 100, 100])
        return jnp.logical_and(jnp.all(-const < x_), jnp.all(x_ < const))

    return jnp.all(jax.vmap(jax.vmap(check_constraints_, in_axes=1))(X))


def check_final_state_constraints(X, const=None):
    if const is None:
        const = jnp.array([0.1, 0.17, 0.1 * 1000, 0.17 * 1000])

    def check_constraints_(x_):
        x_ = mod_pendulum(x_)
        return jnp.logical_and(jnp.all(-const < x_), jnp.all(x_ < const))

    return jnp.all(jax.vmap(jax.vmap(check_constraints_, in_axes=1))(X[:, :, -10:]))


def get_random_init_points(rng_key, num, maxval=None):
    if maxval is None:
        maxval = jnp.array([0.35*0, 3.1415926535, 1.19999886 * 0, 9.99996568 * 0])
    return jax.random.uniform(rng_key, (num, 4), minval=-maxval,
                              maxval=maxval)


def get_simulator(params):
    dt = 160e-3
    params_pertubed = [0.0 for i in range(5)]

    def f_true(x, u):
        return f(x, u, params_pertubed)

    return Simulator(dt, f_true)


class Simulator:
    def __init__(self, timestep, f):
        self.dt = timestep
        self.N = 1

        def f_pwconst_input(y, t, u0):
            x = y
            u = u0
            return f(x, u)

        self.f_pw_const_input = f_pwconst_input

    def run(self, x0, u0):
        X_traj = odeint(self.f_pw_const_input, x0, np.linspace(0, self.dt, self.N + 1), args=tuple([u0], ))
        return X_traj[-1]


def run_tester_round(model, model_aug, normalization, x0, parameter_pertubations):
    dt = 160e-3
    m1 = 0.02

    m2 = m1  # 0.04

    N = int(16 / dt)

    def f_true(x, u):
        return f(x, u, parameter_pertubations)

    sim = Simulator(dt, f_true)

    N_sim = N
    X_cl_wrong_mass = np.zeros((4, N_sim + 1))
    U_cl_wrong_mass = np.zeros((1, N_sim))
    X_cl_lin_corr_mass = np.zeros((4, N_sim + 1))
    U_cl_lin_corr_mass = np.zeros((1, N_sim))

    X_cl_wrong_mass[:, 0] = x0
    X_cl_lin_corr_mass[:, 0] = x0

    for i in range(N_sim):
        U_cl_lin_corr_mass[:, i] = call_control(X_cl_lin_corr_mass[:, i], model, model_aug, normalization,
                                                parameter_pertubations)
        X_cl_lin_corr_mass[:, i + 1] = np.array(sim.run(X_cl_lin_corr_mass[:, i], U_cl_lin_corr_mass[:, i]))

        U_cl_wrong_mass[:, i] = call_control(X_cl_wrong_mass[:, i], model, model_aug, normalization,
                                             parameter_pertubations, augment_input=False)
        X_cl_wrong_mass[:, i + 1] = np.array(sim.run(X_cl_wrong_mass[:, i], U_cl_wrong_mass[:, i]))

    labels = ["AMPC with augmentation", "AMPC without augmentation"]  # , "MPC"]
    U = [U_cl_lin_corr_mass, U_cl_wrong_mass]  # , U]
    X = [X_cl_lin_corr_mass, X_cl_wrong_mass]  # , X]

    return X, U


def run_tester(model, model_aug, normalization, num_init_points=200, saving_path=None):
    m_add_perturbed_max = 0.04 * 0.5 * 2
    ABminusBeq_perturbed_max = 1.0 * 5 * 1.2 * 1.5
    M_perturbed_max = 1.0
    AC_perturbed_max = 0.5 * 2
    B_p_perturbed_max = 0.02

    pertubations_max = [m_add_perturbed_max, ABminusBeq_perturbed_max, M_perturbed_max, AC_perturbed_max, B_p_perturbed_max]

    num_samples = 9

    x_0_samples = get_random_init_points(jax.random.PRNGKey(100), num_init_points, maxval=jnp.array([0.35, 3.1415926535, 0, 0]))
    print(x_0_samples)

    succ_rate_corr = np.array([[0.0 for j in range(num_samples)] for i in range(5)])
    succ_rate_orig = np.array([[0.0 for j in range(num_samples)] for i in range(5)])

    for i in range(0, 5):
        print(i)
        parameter_pertubations = [0, 0, 0, 0, 0]
        k = np.linspace(-1,1,num=num_samples)
        for j in range(0, num_samples):
            parameter_pertubations[i] = pertubations_max[i]*k[j]
            print(parameter_pertubations)
            succ_runs_corr = 0
            succ_runs_orig = 0
            for x0 in x_0_samples:
                X, U = run_tester_round(model, model_aug, normalization, x0, parameter_pertubations)
                if check_final_state_constraints(jnp.array([X[0]]), const=jnp.array([0.1, 0.17, 0.1 * 1000, 0.17 * 1000])) and check_constraints(jnp.array([X[0]])):
                    succ_runs_corr += 1
                elif j == 4:
                    dt = 160e-3
                    N_sim = 100
                    plot_pendulum(np.linspace(0, dt * N_sim, N_sim + 1), 9, U, X, ["a", "b"], latexify=False)
                if check_final_state_constraints(jnp.array([X[1]]), const=jnp.array([0.1, 0.17, 0.1 * 1000, 0.17 * 1000])) and check_constraints(jnp.array([X[1]])):
                    succ_runs_orig += 1
            succ_rate_corr[i][j] = succ_runs_corr / len(x_0_samples)
            succ_rate_orig[i][j] = succ_runs_orig / len(x_0_samples)

    plot_heatmap(np.linspace(-1,1,num=num_samples), succ_rate_corr, succ_rate_orig)
    print(succ_rate_corr)
    print(succ_rate_orig)

    if saving_path is None:
        np.savetxt("/home/alex/Downloads/succ_rate_corr.txt", succ_rate_corr)
        np.savetxt("/home/alex/Downloads/succ_rate_orig.txt", succ_rate_orig)
    else:
        np.savetxt(f"{saving_path}/succ_rate_corr.txt", succ_rate_corr)
        np.savetxt(f"{saving_path}/succ_rate_orig.txt", succ_rate_orig)


def run(model_path, iteration, num_neurons_contr, num_layers_contr, num_neurons_aug, num_layers_aug, num_init_points):
    path = f"{model_path}/model_{num_layers_contr}x{num_neurons_contr}/It{iteration}"
    path_aug = f"{model_path}/model_{num_layers_aug}x{num_neurons_aug}_aug/It{iteration}"
    parameter_path = path + "/params.yaml"
    with open(parameter_path, "r") as file:
        params = yaml.safe_load(file)

    normalization_path = path + "/normalization.p"
    with open(normalization_path, "rb") as file:
        normalization = pickle.load(file)
    init_key = jax.random.PRNGKey(1)
    model = AMPCNN(num_layers=params["num_layers"], num_neurons=params["num_neurons"],
                   num_sys_states=params["num_sys_states"], num_sys_inputs=params["num_sys_inputs"],
                   num_aug_params=params["num_aug_params"], rng_key=init_key,
                   activation_function=params["activation_function"])
    model = model.load_model_from_file(path)

    parameter_path_aug = path_aug + "/params.yaml"
    with open(parameter_path_aug, "r") as file:
        params_aug = yaml.safe_load(file)

    normalization_path = path_aug + "/normalization.p"
    with open(normalization_path, "rb") as file:
        normalization_aug = pickle.load(file)
        normalization["gradient"] = normalization_aug["gradient"]
    model_aug = AMPCAUGNN(num_layers=params_aug["num_layers"], num_neurons=params_aug["num_neurons"],
                          num_sys_states=params_aug["num_sys_states"], num_sys_inputs=params_aug["num_sys_inputs"],
                          num_aug_params=params_aug["num_aug_params"], rng_key=init_key,
                          activation_function=params_aug["activation_function"])
    model_aug = model_aug.load_model_from_file(path_aug)

    run_tester(model, model_aug, normalization, num_init_points=num_init_points, saving_path=path)


def plot_trials(model_path, num_neurons_contr, num_layers_contr, num_samples=9):
    for iteration in range(11):
        path = f"{model_path}/model_{num_layers_contr}x{num_neurons_contr}/It{iteration}"
        succ_rate_corr = np.loadtxt(f"{path}/succ_rate_corr.txt")
        succ_rate_orig = np.loadtxt(f"{path}/succ_rate_orig.txt")
        plot_heatmap(np.linspace(-1, 1, num=num_samples), succ_rate_corr, succ_rate_orig)

if __name__ == "__main__":
    #plot_trials(model_path="/home/alex/hpc_data/pendulum_swingup30/trainer/",
    #            num_neurons_contr=50, num_layers_contr=5, num_samples=9)
    #exit(0)

    #path = "/home/alex/hpc_data/pendulum_swingup15V2/trainer/model_5x50/It0"
    #path_aug = "/home/alex/hpc_data/pendulum_swingup15V2/trainer/model_8x50_aug/It0"
    #path = "/home/alex/hpc_data/pendulum_swingup15/trainer/model_5x50/It3"
    #path_aug = "/home/alex/hpc_data/pendulum_swingup15/trainer/model_8x50_aug/It3"
    path = "/home/alex/hpc_data/pendulum_swingupNew15/trainer/model_5x50/It0"
    path_aug = "/home/alex/hpc_data/pendulum_swingupNew15/trainer/model_8x50_aug/It0"
    jax.config.update('jax_platform_name', 'cpu')
    parameter_path = path + "/params.yaml"
    with open(parameter_path, "r") as file:
        params = yaml.safe_load(file)

    normalization_path = path + "/normalization.p"
    with open(normalization_path, "rb") as file:
        normalization = pickle.load(file)
    print(normalization)
    # dataset = AMPCDataset(params)
    init_key = jax.random.PRNGKey(1)
    model = AMPCNN(num_layers=params["num_layers"], num_neurons=params["num_neurons"],
                   num_sys_states=params["num_sys_states"], num_sys_inputs=params["num_sys_inputs"],
                   num_aug_params=params["num_aug_params"], rng_key=init_key,
                   activation_function=params["activation_function"])
    model = model.load_model_from_file(path)

    parameter_path_aug = path_aug + "/params.yaml"
    with open(parameter_path_aug, "r") as file:
        params_aug = yaml.safe_load(file)

    normalization_path = path_aug + "/normalization.p"
    with open(normalization_path, "rb") as file:
        normalization_aug = pickle.load(file)
        print(normalization_aug["gradient"])
        print(normalization["gradient"])
        normalization["gradient"] = normalization_aug["gradient"]
    model_aug = AMPCAUGNN(num_layers=params_aug["num_layers"], num_neurons=params_aug["num_neurons"],
                   num_sys_states=params_aug["num_sys_states"], num_sys_inputs=params_aug["num_sys_inputs"],
                   num_aug_params=params_aug["num_aug_params"], rng_key=init_key,
                   activation_function=params_aug["activation_function"])
    model_aug = model_aug.load_model_from_file(path_aug)

    run_tester(model, model_aug, normalization, num_init_points=10)

    exit(0)

    # for i, data in enumerate(dataset.eval_ds):
        # print(f"{model_aug(data['sys_state'])}, {data['params_aug_gradient']}")

    dt = 50e-3 #160e-3
    m1 = 0.02
    m_add_perturbed = (-0.02 * 0.0)
    ABminusBeq_perturbed = -1.0 * 0.0
    M_perturbed = 1.0 * 0.0
    AC_perturbed = 0.0
    B_p_perturbed = 0.01 * 0

    parameter_pertubations = [m_add_perturbed, ABminusBeq_perturbed, M_perturbed, AC_perturbed, B_p_perturbed]

    m2 = m1 #0.04
    print(m2)

    N = int(10 / dt)

    # N = 100

    x0_flat = np.array([0.35*0, 3.14159, 0.0, 0.0])# np.array([-0.029150275832573513, -2.6582845139655964, -0.5333006949571231, -7.5397636927576634])
    U_flat = np.array(
        [-3.1990419294028176, -0.794773068138816, 1.2531017316535817, -0.22106537511282393, -3.1747048283292076,
         -0.5857184075490766, 7.702761812426266, 2.2829876722361386, -1.1493895079346645, -1.0061765988246132,
         -0.4835169568239018, -0.18159892019485632, -0.0417152079345123, 0.017508910825344986, 0.03932434657693387,
         0.0443058601839241, 0.04205987631319994, 0.037011734420273334, 0.031144341578844624, 0.025293193048218046,
         0.01975798896624898, 0.014589296916778889, 0.00972540994894956, 0.0051137789255530095, 0.0013242484134987592])
    X_flat = np.array(
        [-0.029150275832573513, -2.6582845139655964, -0.5333006949571231, -7.5397636927576634, -0.11567812737956844,
         -3.8820692547321083, -0.3710871218320493, -6.876666307730988, -0.12962957960070415, -4.7316242960277135,
         0.029775254595346058, -3.808039767179991, -0.10893270168684664, -5.112972156082754, 0.15721719777894766,
         -0.9890674833859736, -0.1026652742196383, -5.040252273358829, -0.02909383507857863, 1.8716022910006231,
         -0.1373518792686939, -4.496297506202736, -0.279619873884547, 4.88365576767026, -0.14001609605106005,
         -3.479704659952358, 0.153098437017297, 7.496003057076204, -0.038359802402083955, -2.159042589369665,
         0.7243284454473713, 7.688780381103804, 0.01885072832539487, -1.1899986120827508, 0.21606045148444042,
         4.586951268866338, 0.020771869626458304, -0.6187834407223295, -0.0866361996890321, 2.6527776023707665,
         0.005291698087658117, -0.3043719002631102, -0.10378441688506705, 1.41145438672515, -0.006078202464662078,
         -0.14468580277423576, -0.05601488889690508, 0.6926788778954206, -0.011243703105757558, -0.06795001313271198,
         -0.020925626307421424, 0.3268575742083951, -0.012649737166308978, -0.03200328585425957, -0.0029343821460961626,
         0.15181291428994034, -0.012266114153331967, -0.015324807197089379, 0.00492621250886948, 0.07008603650834536,
         -0.011166655096152485, -0.007597850296244257, 0.007728654960763801, 0.03232519080785942, -0.009870279858225263,
         -0.003999040762286438, 0.008183332142653363, 0.014971816178419015, -0.00861045650662248,
         -0.0022979263488892847, 0.007621738790239718, 0.007018754335934126, -0.007482644902724832,
         -0.0014694464758818356, 0.006673741125538314, 0.0033730536598768133, -0.006518088525413136,
         -0.001044300815063289, 0.005626117171145289, 0.0016950118046194676, -0.005719699784808365,
         -0.0008074049831362683, 0.004600435351791135, 0.0009205583980546864, -0.005079296086279481,
         -0.0006581466650591143, 0.0036393588248536265, 0.0005789347116663735, -0.004585844403593736,
         -0.0005446649604291179, 0.0027477779321710997, 0.0004852832316331228, -0.004229420199639719,
         -0.0004305533987177648, 0.0019123024164723473, 0.0006136613722113719, -0.004002448832471721,
         -0.0002745236357741861, 0.0011172570797733906, 0.0010528184328218197, -0.0038919295054432234,
         -2.5453422734502465e-05, 0.0004257866070094434, 0.0018798384458440195])
    J_flat = np.array(
        [110.42603032516227, 0.28468080111658445, 1.0215166651082577, 2.547775855859885, -186.83094321289474,
         62.82169639468288, 0.19305889133863585, 0.6377428446848278, 1.2223859331939986, -73.87296444407477,
         -30.405503565954287, -0.10863050901628601, 0.20765997438780387, -0.8135831969627296, 78.72701799782392,
         -6.478083456644055, -0.10518873517270577, -0.23915990904970102, 0.0040907920278002155, -3.7747767594061563,
         11.098480940571067, 0.2282962566561193, -0.07950766521104492, 2.4775410309351993, -169.80432395134957,
         -211.93425013532357, 0.3299403235495759, -0.3581286273768403, 1.6714600302348561, 12.625620342269185,
         -65.25634247218203, -0.48013131535236475, -0.5044847206504778, -4.971110094204886, 336.9912928810636,
         200.2881566290515, -0.3887572083870632, 1.0763866741417785, -3.053345425559211, 30.183870387016647,
         18.723297378403284, -0.04844918463633108, 0.028457183563057604, 0.12262270370770728, -34.580092899248555,
         -30.972857395878233, 0.03458837663959513, 0.07848969503742477, 0.5286713266632036, -16.299677142477254,
         -24.479596147686827, 0.026919554335237128, 0.12443136808840574, 0.2899624294714738, -3.3675588868573354,
         -15.653700725240354, 0.009526822159472564, 0.09979665212849934, 0.09578494748245263, 2.1485037857681895,
         -9.503130691939155, -0.0016289587091675717, 0.06897028665958164, -0.003609490284257517, 3.985761490300411,
         -6.037248843855839, -0.007057729725804741, 0.048649147322906094, -0.046826267139315404, 4.310259422459482,
         -4.049161494199547, -0.00910402599348567, 0.03680120336444632, -0.06195287356641376, 4.040723985345001,
         -2.8596908858637557, -0.009393450887257258, 0.029570654499966605, -0.06373186060748422, 3.561515829072784,
         -2.1071430931650705, -0.008833025239125693, 0.02455789295301284, -0.05944003931788378, 3.035884647254655,
         -1.5990276534754915, -0.007894229630679192, 0.02055781608795861, -0.052602631599564395, 2.530354150343859,
         -1.2319637564978443, -0.006813040691846246, 0.01700879375560665, -0.044891139033266275, 2.0679999443464876,
         -0.9494197942655285, -0.005702598467926888, 0.013655233100745536, -0.037062983002739025, 1.652531246055341,
         -0.7194297695234865, -0.0046127228088715805, 0.01037922900507249, -0.029425198799507692, 1.2792631096835267,
         -0.5228244857630977, -0.00356034064803408, 0.007127117993768006, -0.022064548148400007, 0.9399652991592927,
         -0.347086112828283, -0.002543485734719613, 0.0039098263296339685, -0.014975519834762889, 0.6251328307935021,
         -0.18475193996514397, -0.0015455088748743814, 0.0009654882272010312, -0.008221872465404711, 0.3286282300565788,
         -0.05035074069337813, -0.0005920174883263588, -0.0008519816319100466, -0.0025278627280065787,
         0.08551173571213128])

    x0 = x0_flat
    #U = U_flat.reshape((N, 1)).transpose()
    #X = X_flat.reshape((N + 1, 4)).transpose()
    #J = J_flat.reshape((N, 5)).transpose()

    parameter_pertubations_sim = [m_add_perturbed, ABminusBeq_perturbed, M_perturbed, AC_perturbed, B_p_perturbed]
    def f_true(x, u):
        return f(x, u, parameter_pertubations_sim)


    sim = Simulator(dt, f_true)

    N_sim = N
    X_cl_wrong_mass = np.zeros((4, N_sim + 1))
    U_cl_wrong_mass = np.zeros((1, N_sim))
    X_cl_lin_corr_mass = np.zeros((4, N_sim + 1))
    U_cl_lin_corr_mass = np.zeros((1, N_sim))

    X_cl_wrong_mass[:, 0] = x0
    X_cl_lin_corr_mass[:, 0] = x0

    for i in range(N_sim):
        U_cl_lin_corr_mass[:, i] = call_control(X_cl_lin_corr_mass[:, i], model, model_aug, normalization, parameter_pertubations)
        X_cl_lin_corr_mass[:, i + 1] = np.array(sim.run(X_cl_lin_corr_mass[:, i], U_cl_lin_corr_mass[:, i]))

        U_cl_wrong_mass[:, i] = call_control(X_cl_wrong_mass[:, i], model, model_aug, normalization, parameter_pertubations, augment_input=False)
        X_cl_wrong_mass[:, i + 1] = np.array(sim.run(X_cl_wrong_mass[:, i], U_cl_wrong_mass[:, i]))

    labels = ["AMPC with augmentation", "AMPC without augmentation"] #, "MPC"]
    print(check_final_state_constraints(jnp.array([X_cl_lin_corr_mass])))
    print(check_constraints(jnp.array([X_cl_lin_corr_mass])))

    # X_cl_wrong_mass = simulate_plant(jnp.array([x0]), get_simulator(None), model, normalization, params, model_aug=None)
    U = [U_cl_lin_corr_mass, U_cl_wrong_mass[0]] #, U]
    X = [X_cl_lin_corr_mass, X_cl_wrong_mass] #, X]
    plot_pendulum(np.linspace(0, dt * N_sim, N_sim + 1), 9, U, X, labels, latexify=False)

    """test_dl = utils.get_dataloader(dataset.test_ds,
                                   batch_size=1,
                                   num_workers=params["dataloader_num_workers"])
    for batch_nr, batch in tqdm(enumerate(test_dl)):
        print(f"{jax.vmap(model)(batch['sys_state'].numpy())}, {batch['sys_input'].numpy()}")"""


