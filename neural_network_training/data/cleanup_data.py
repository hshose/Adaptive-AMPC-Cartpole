import numpy as np
import jax.numpy as jnp
import jax

def cleanup(params):
    # load data
    x0_data = jnp.array(np.genfromtxt(f"{params['dataset_path']}/x0.csv", delimiter=','))
    print("Loaded x0")

    u_data = jnp.array(np.genfromtxt(f"{params['dataset_path']}/U.csv", delimiter=',')[:, 0:params["num_sys_inputs"]])
    print("Loaded u")

    gradient_data = jnp.array(np.genfromtxt(
        f"{params['dataset_path']}/J.csv", delimiter=',')[:, 0:params["num_sys_inputs"] * params["num_aug_params"]])
    print("loaded gradient")

