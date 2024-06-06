import numpy as np
import jax.numpy as jnp
import jax
import yaml
import pickle

from jinja2 import Template, Environment, FileSystemLoader

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from neural_networks.jax_models import AMPCNN, AMPCAUGNN, MLP
from data.dataset import AMPCDataset, normalize, denormalize




def generate_tanh_lookup():
    step_size = 0.01
    max_value = 5.0
    a = np.array([-max_value + step_size*i for i in range(int(round(2*max_value / step_size)))])
    c = np.tanh(a)
    for e in c:
        print(e, end=",")

    print()
    print(len(c))


def generate_model_struct_u(name, path):
    print(f"Loading model {name} from {path}...")
    jax.config.update('jax_platform_name', 'cpu')
    parameter_path = path + "/params.yaml"
    with open(parameter_path, "r") as file:
        params = yaml.safe_load(file)
    print(f"{params=}")

    normalization_path = path + "/normalization.p"
    with open(normalization_path, "rb") as file:
        normalization = pickle.load(file)
    print(f"{normalization=}")

    
    init_key = jax.random.PRNGKey(1)
    model = AMPCNN(num_layers=params["num_layers"], num_neurons=params["num_neurons"],
                   num_sys_states=params["num_sys_states"], num_sys_inputs=params["num_sys_inputs"],
                   num_aug_params=params["num_aug_params"], rng_key=init_key,
                   activation_function=params["activation_function"])
    model = model.load_model_from_file(path)

    print(denormalize(model(normalize(jnp.array([-1.58333e-01,1.22718e-02,-1.68788e-01,3.12718e-02]), normalization["x"])), normalization['u']))
    
    return {
        'name':          str(name),
        'input_size':    params["num_sys_states"],
        'input_offset':  MLP.generate_matrix_code(normalization['x'][0]),
        'input_scale':   MLP.generate_matrix_code(normalization['x'][1]),
        'output_size':   params["num_sys_inputs"],
        'output_offset': MLP.generate_matrix_code(normalization['u'][0]),
        'output_scale':  MLP.generate_matrix_code(normalization['u'][1]),
        'layer_list':    model.generate_code()
    }


def generate_model_struct_J(name, path):
    print(f"Loading model {name} from {path}...")
    jax.config.update('jax_platform_name', 'cpu')
    parameter_path = path + "/params.yaml"
    with open(parameter_path, "r") as file:
        params = yaml.safe_load(file)
    print(f"{params=}")

    normalization_path = path + "/normalization.p"
    with open(normalization_path, "rb") as file:
        normalization = pickle.load(file)
    print(f"{normalization=}")

    
    init_key = jax.random.PRNGKey(1)
    model = AMPCAUGNN(num_layers=params["num_layers"], num_neurons=params["num_neurons"],
                   num_sys_states=params["num_sys_states"], num_sys_inputs=params["num_sys_inputs"],
                   num_aug_params=params["num_aug_params"], rng_key=init_key,
                   activation_function=params["activation_function"])
    model = model.load_model_from_file(path)

    print(denormalize(model(normalize(jnp.array([0.0, 0.0, 0.0, 0.0]), normalization["x"])), normalization['gradient']))

    
    return {
        'name':          str(name),
        'input_size':    model.num_sys_states,
        'input_offset':  MLP.generate_matrix_code(normalization['x'][0]),
        'input_scale':   MLP.generate_matrix_code(normalization['x'][1]),
        'output_size':   model.output_dim,
        'output_offset': MLP.generate_matrix_code(normalization['gradient'][0]),
        'output_scale':  MLP.generate_matrix_code(normalization['gradient'][1]),
        'layer_list':    model.generate_code()
    }
    

def generate_cpp():
    
    jinja_environment = Environment(loader=FileSystemLoader('./templates'))
    template_hpp = jinja_environment.get_template('neural_network.hpp.jinja')
    template_cpp = jinja_environment.get_template('neural_network.cpp.jinja')

    neural_network_u = generate_model_struct_u("u", '../models/model_8x50/It3')
    neural_network_J = generate_model_struct_J("J", '../models/model_10x50_aug/It3')

    template_values = {'network_list': [neural_network_u, neural_network_J]}

    output = template_hpp.render(template_values)
    with open('./neural_network.hpp', 'w') as f:
        f.write(output)

    output = template_cpp.render(template_values)
    with open('./neural_network.cpp', 'w') as f:
        f.write(output)


if __name__ == "__main__":
    generate_cpp()
