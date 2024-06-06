from jinja2 import Template, Environment, FileSystemLoader

import numpy as np

jinja_environment = Environment(loader=FileSystemLoader('templates'))
template_hpp = jinja_environment.get_template('neural_network.hpp.jinja')     
template_cpp = jinja_environment.get_template('neural_network.cpp.jinja')     

x_offset = np.random.rand(4)
x_scale = np.random.rand(4)
u_offset = np.random.rand(1)
u_scale = np.random.rand(1)

J_offset = np.random.rand(5)
J_scale = np.random.rand(5)

layer0_weights = np.random.rand(4,50)
layer0_bias = np.random.rand(50)

layer1_weights = np.random.rand(50,1)
layer1_bias = np.random.rand(1)

layer2_weights = np.random.rand(50,5)
layer2_bias = np.random.rand(5)


def np2cpp( arr ):
    if arr.ndim == 1:
        return '{'+', '.join(str(num) for num in arr)+'}'
    if arr.ndim == 2:
        return '{'+','.join('{'+','.join(str(num) for num in arr[:,i])+'}' for i in range(len(arr[0,:]))) + '}'
        
    

layer0 = {
    'activation': 'TANH',
    'input_size': 4,
    'output_size': 50,
    'weights': np2cpp(layer0_weights),
    'bias': np2cpp(layer0_bias)
}

layer1 = {
    'activation': 'LINEAR',
    'input_size': 50,
    'output_size': 1,
    'weights': np2cpp(layer1_weights),
    'bias': np2cpp(layer1_bias)
}

layer2 = {
    'activation': 'LINEAR',
    'input_size': 50,
    'output_size': 5,
    'weights': np2cpp(layer2_weights),
    'bias': np2cpp(layer2_bias)
}


neural_network_u = {
    'name':          'u',
    'input_size':    4,
    'input_offset':  np2cpp(x_offset),
    'input_scale':   np2cpp(x_scale),
    'output_size':   1,
    'output_offset': np2cpp(u_offset),
    'output_scale':  np2cpp(u_scale),
    'layer_list':    [layer0, layer1]
}

neural_network_J = {
    'name':          'J',
    'input_size':    4,
    'input_offset':  np2cpp(x_offset),
    'input_scale':   np2cpp(x_scale),
    'output_size':   5,
    'output_offset': np2cpp(J_offset),
    'output_scale':  np2cpp(J_scale),
    'layer_list':    [layer0, layer2]
}

template_values = {'network_list': [neural_network_u, neural_network_J]}


output = template_hpp.render(template_values)
with open('neural_network.hpp', 'w') as f:
    f.write(output)

output = template_cpp.render(template_values)
with open('neural_network.cpp', 'w') as f:
    f.write(output)