import jax.numpy as jnp
import jax
import equinox as eqx
import math

import yaml
import pickle

from neural_networks.jax_layers import FullyConnectedLayer


class MLP(eqx.Module):
    layers: list

    def __init__(self, layer_structure, rng_key, activation_function='linear'):
        """

        Parameters
        ----------
            layer_structure: list(int) list with the layer structure (how many neurons per layer)
        """
        super().__init__()

        if len(layer_structure) < 2:
            self.layers = []
        else:
            keys = jax.random.split(rng_key, num=len(layer_structure) - 1)
            self.layers = [FullyConnectedLayer(input_dim=layer_structure[0], output_dim=layer_structure[1], rng_key=keys[0],
                                               activation_function=activation_function)]

            # add activation functions and layers as per "units" input
            for i in range(1, len(layer_structure) - 1):
                self.layers.append(FullyConnectedLayer(input_dim=layer_structure[i], output_dim=layer_structure[i+1],
                                                       rng_key=keys[i],
                                                       activation_function=activation_function if i < len(layer_structure) - 2 else "linear"))

    def __call__(self, x):
        def call_mlp(x_):
            for idx, layer in enumerate(self.layers):
                x_ = layer(x_)
            else:
                return x_

        if len(x.shape) == 2:
            mlp_vmap_batch = jax.vmap(call_mlp)
            return mlp_vmap_batch(x)
        return call_mlp(x)

    def generate_code(self):
        layer_template_values = []
        for idx, layer in enumerate(self.layers):
            template_values = {'activation': layer.activation_function_name.upper(),
                               'input_size': layer.input_dim,
                               'output_size': layer.output_dim,
                               'weights': MLP.generate_matrix_code(layer.weight),
                               'bias': MLP.generate_matrix_code(layer.bias)
                               }
            layer_template_values.append(template_values)
        return layer_template_values

    @staticmethod
    def generate_matrix_code(matrix):
        data = "{"
        if len(matrix.shape) == 1:
            for i in range(len(matrix)):
                data += f"{float(matrix[i])}, "
        else:
            for i in range(len(matrix)):
                data += f"{MLP.generate_matrix_code(matrix[i])}, "
        return data[0:-1] + "}"


class AMPCNN(eqx.Module):
    output_dim: int
    num_sys_states: int
    num_sys_inputs: int
    num_aug_params: int
    neural_network: any

    def __init__(self, num_layers, num_neurons, num_sys_states, num_sys_inputs, num_aug_params, rng_key,
                 activation_function="tanh"):
        super().__init__()
        self.num_sys_states = num_sys_states
        self.num_sys_inputs = num_sys_inputs
        self.num_aug_params = num_aug_params
        self.output_dim = num_sys_inputs  # + num_aug_params * num_sys_inputs
        layer_structure = [num_sys_states]
        for i in range(num_layers-1):
            layer_structure.append(num_neurons)
        layer_structure.append(self.output_dim)

        self.neural_network = MLP(layer_structure, rng_key, activation_function=activation_function)

    @eqx.filter_jit
    def __call__(self, x):
        return self.neural_network(x)

    def save_model_to_file(self, path):
        eqx.tree_serialise_leaves(f"{path}/model.eqx", self)

    def save_model_to_file_temp(self, path):
        eqx.tree_serialise_leaves(f"{path}/model_temp.eqx", self)

    def load_model_from_file(self, path):
        return eqx.tree_deserialise_leaves(f"{path}/model.eqx", self)

    def load_model_from_file_temp(self, path):
        return eqx.tree_deserialise_leaves(f"{path}/model_temp.eqx", self)

    def generate_code(self):
        return self.neural_network.generate_code()


class AMPCAUGNN(eqx.Module):
    output_dim: int
    num_sys_states: int
    num_sys_inputs: int
    num_aug_params: int
    neural_network: any

    def __init__(self, num_layers, num_neurons, num_sys_states, num_sys_inputs, num_aug_params, rng_key,
                 activation_function="tanh"):
        super().__init__()
        self.num_sys_states = num_sys_states
        self.num_sys_inputs = num_sys_inputs
        self.num_aug_params = num_aug_params
        self.output_dim = num_aug_params * num_sys_inputs
        layer_structure = [num_sys_states]
        for i in range(num_layers-1):
            layer_structure.append(num_neurons)
        layer_structure.append(self.output_dim)

        self.neural_network = MLP(layer_structure, rng_key, activation_function=activation_function)

    @eqx.filter_jit
    def __call__(self, x):
        y = self.neural_network(x)
        params_aug_gradient = jnp.reshape(y, (self.num_sys_inputs, self.num_aug_params))

        return params_aug_gradient

    def save_model_to_file(self, path):
        eqx.tree_serialise_leaves(f"{path}/model_aug.eqx", self)

    def save_model_to_file_temp(self, path):
        eqx.tree_serialise_leaves(f"{path}/model_aug_temp.eqx", self)

    def load_model_from_file(self, path):
        return eqx.tree_deserialise_leaves(f"{path}/model_aug.eqx", self)

    def generate_code(self):
        return self.neural_network.generate_code()


if __name__ == "__main__":
    path = "/home/alex/hpc_data/pendulum_swingup/trainer/model_5x100/It0"
    saving_path = "../embedded_nn_inference/"
    jax.config.update('jax_platform_name', 'cpu')
    parameter_path = path + "/params.yaml"
    with open(parameter_path, "r") as file:
        params = yaml.safe_load(file)

    normalization_path = path + "/normalization.p"
    with open(normalization_path, "rb") as file:
        normalization = pickle.load(file)
    # dataset = AMPCDataset(params)
    init_key = jax.random.PRNGKey(1)
    model = AMPCNN(num_layers=params["num_layers"], num_neurons=params["num_neurons"],
                   num_sys_states=params["num_sys_states"], num_sys_inputs=params["num_sys_inputs"],
                   num_aug_params=params["num_aug_params"], rng_key=init_key,
                   activation_function=params["activation_function"])
    model = model.load_model_from_file(path)

    print(model(jnp.array([0.0, 0.0, 0.0, 0.0])))

    static_code, c_code = model.generate_code()

    with open(saving_path + "neural_network.hpp", 'w') as f:
        f.write(static_code)

    with open(saving_path + "neural_network.cpp", 'w') as f:
        f.write(c_code)


