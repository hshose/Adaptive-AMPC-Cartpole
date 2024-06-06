from torch.utils.data import Dataset
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import yaml

import jax

from neural_networks.jax_models import AMPCNN, AMPCAUGNN

# from tester import run_testing

import pickle

def normalize(data, normalization_params):
    return (data - normalization_params[0]) / normalization_params[1]


def denormalize(data, normalization_params):
    return data * normalization_params[1] + normalization_params[0]


class AMPCPartDataset(Dataset):
    """MLP Dataset class. For eval, trainer and test"""
    def __init__(self, sys_state, sys_input, params_aug_gradient):
        self.sys_state = sys_state
        self.sys_input = sys_input
        self.params_aug_gradient = params_aug_gradient

    def __len__(self):
        return len(self.sys_state)

    def __getitem__(self, idx):
        return {'sys_state': self.sys_state[idx], 'sys_input': self.sys_input[idx], "params_aug_gradient": self.params_aug_gradient[idx]}


class AMPCDataset(Dataset):

    def __init__(self, params):
        # load data
        x0_data = np.genfromtxt(f"{params['dataset_path']}/x0.csv", delimiter=',')
        self.x0_data = x0_data
        print("Loaded x0")

        """x_data = np.genfromtxt(f"{params['dataset_path']}/X.csv", delimiter=',')
        print(np.min(x_data, axis=0))
        print(np.max(x_data, axis=0))"""

        u_data = np.genfromtxt(f"{params['dataset_path']}/U.csv", delimiter=',')[:, 0:params["num_sys_inputs"]]
        self.u_data = u_data
        print("Loaded u")

        gradient_data = np.genfromtxt(
            f"{params['dataset_path']}/J.csv", delimiter=',')[:, 0:params["num_sys_inputs"]*params["num_aug_params"]]
        self.gradient_data = gradient_data
        print("loaded gradient")

        size_dataset = len(x0_data)

        # shuffle data
        np.random.seed(1)
        shuffle_vec = np.array([i for i in range(size_dataset)])
        np.random.shuffle(shuffle_vec)
        x0_data = x0_data[shuffle_vec, :]
        u_data = u_data[shuffle_vec, :]
        gradient_data = np.reshape(gradient_data[shuffle_vec, :],
                                   (size_dataset, params["num_sys_inputs"], params["num_aug_params"]))

        self.normalization = {}
        self.normalization["x"] = (np.mean(x0_data, axis=0), np.max(np.abs(x0_data), axis=0))
        self.normalization["u"] = (np.mean(u_data, axis=0), np.max(np.abs(u_data), axis=0))
        self.normalization["gradient"] = (np.mean(gradient_data, axis=0), 3*np.std(np.abs(gradient_data), axis=0))

        x0_data = normalize(x0_data, self.normalization["x"])
        u_data = normalize(u_data, self.normalization["u"])
        gradient_data = np.clip(normalize(gradient_data, self.normalization["gradient"]), a_min=-2, a_max=2)
        #print(np.max(gradient_data, axis=0))
        #print(np.min(gradient_data, axis=0))

        """for i in range(5):
            plt.hist(gradient_data[:, 0, i], bins=1000)
            plt.show()"""

        # assert False

        size_training = int(round(size_dataset * params["train_size"]))
        size_eval = int(round(size_dataset * params["evaluation_size"]))

        self.train_ds = AMPCPartDataset(x0_data[0:size_training, :], u_data[0:size_training, :], gradient_data[0:size_training, :])
        self.eval_ds = AMPCPartDataset(x0_data[size_training:size_training + size_eval, :],
                                       u_data[size_training:size_training + size_eval, :],
                                       gradient_data[size_training:size_training + size_eval, :])
        self.test_ds = AMPCPartDataset(x0_data[size_training + size_eval:, :],
                                       u_data[size_training + size_eval:, :],
                                       gradient_data[size_training + size_eval:, :])

        self.num_sys_states = params["num_sys_states"]
        self.num_sys_inputs = params["num_sys_inputs"]
        self.num_aug_params = params["num_aug_params"]


def add_zeros():
    path = "/home/alex/torch_datasets/AMPCPendulum15"
    path0 = "/home/alex/torch_datasets/AMPCPendulum100"
    path_saving = "/home/alex/torch_datasets/AMPCPendulum30"

    percentage = 0.15e-2
    x0_data = np.genfromtxt(f"{path}/x0.csv", delimiter=',')
    print("Loaded x0")

    """x_data = np.genfromtxt(f"{params['dataset_path']}/X.csv", delimiter=',')
    print(np.min(x_data, axis=0))
    print(np.max(x_data, axis=0))"""

    u_data = np.genfromtxt(f"{path}/U.csv", delimiter=',')
    print("Loaded u")

    gradient_data = np.genfromtxt(
        f"{path}/J.csv", delimiter=',')[:, 0:1 * 5]
    print("loaded gradient")

    ################################################################
    size = int(percentage * len(x0_data))
    print(size)

    x0_data0 = np.genfromtxt(f"{path0}/x0.csv", delimiter=',')[-size:]
    print("Loaded x0")

    u_data0 = np.genfromtxt(f"{path0}/U.csv", delimiter=',')[-size:]
    print("Loaded u")

    gradient_data0 = np.genfromtxt(
        f"{path0}/J.csv", delimiter=',')[-size:, 0:1 * 5]
    print("loaded gradient")

    np.savetxt(f"{path_saving}/x0.csv", np.concatenate((x0_data, x0_data0)), delimiter=',')
    np.savetxt(f"{path_saving}/U.csv", np.concatenate((u_data, u_data0)), delimiter=',')
    np.savetxt(f"{path_saving}/J.csv", np.concatenate((gradient_data, gradient_data0)), delimiter=',')



if __name__ == "__main__":
    #add_zeros()
    #exit(0)

    parameter_path = "../parameters/pendulum.yaml"
    with open(parameter_path, "r") as file:
        params = yaml.safe_load(file)

    dataset = AMPCDataset(params)

    selected = np.all(np.abs(dataset.x0_data[:, 2:4]) < 1e-1, axis=1)
    x = dataset.x0_data[selected][:, 0]
    y = dataset.x0_data[selected][:, 1]
    z = dataset.u_data[selected][:, 0]

    path = "/home/alex/hpc_data/pendulum_swingup/trainer/model_5x50/It10"
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

    X = np.concatenate([np.array([x]).T, np.array([y]).T, np.ones((len(x), 2))*0], axis=1)

    def call_control_wrapper(x_):
        return run_testing.call_control(x_, model, None, normalization, params["add_weight"], augment_input=False)

    call_control_v = jax.vmap(call_control_wrapper)

    Z = call_control_v(X)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.scatter(x, y, z, c=z, cmap='coolwarm') #, size = 30)
    # ax.scatter(x, y, Z, color="r")

    ax.set_xlabel('y')
    ax.set_ylabel('angle')
    ax.set_zlabel('u')

    plt.show()

