import jax.numpy as jnp
import jax
import equinox as eqx
import math


def init_weight(dim_in, dim_out, key):
    stdv = 1. / math.sqrt(dim_out)
    return jax.random.uniform(key, (dim_out, dim_in)) * 2 * stdv - stdv


def init_bias(dim_out, key):
    stdv = 1. / math.sqrt(dim_out)
    return jax.random.uniform(key, (dim_out,)) * 2 * stdv - stdv


class FullyConnectedLayer(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    activation: callable
    activation_function_name: str
    input_dim: int
    output_dim: int

    # mask: any

    def __init__(self, input_dim, output_dim, rng_key, activation_function='linear'):
        """

        Parameters
        ----------
            input_dim: int
            output_dim: int
            num_nodes: int
            dropout_prob_mode1: float
            dropout_prob_mode2: float
            dropout_prob_mode3: float
            num_nodes: int number of nodes in the network
            num_bits: int number of bit during quantization
            use_interdevice_pruning: bool if inter-device pruning or normal neuron wise pruning should be used.
            activation_function: str 'linear' or 'relu'. This selects the activation function of the network.
        """
        super().__init__()

        key_weight, key_bias = jax.random.split(rng_key)
        self.weight = init_weight(input_dim, output_dim, key_weight)
        self.bias = init_bias(output_dim, key_bias)

        self.input_dim = input_dim
        self.output_dim = output_dim

        if activation_function == 'linear':
            self.activation = None
        elif activation_function == 'relu':
            self.activation = jax.nn.relu
        elif activation_function == 'tanh':
            self.activation = jax.nn.tanh

        self.activation_function_name = activation_function

    def __call__(self, x):
        y = self.weight @ x + self.bias

        if self.activation is None:
            return y
        else:
            return self.activation(y)

