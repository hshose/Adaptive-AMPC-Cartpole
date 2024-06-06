#include <iostream>
#include <array>
#include <string>

#include "fc_layer.hpp"
#include "activation_function.hpp"
#include "neural_network.hpp"

using namespace std;
using namespace embedded_nn_inference;

int main()
{
    array<float, 4> input = normalize_x({0.0, 0.0, 0.0, 0.0});
    array<float, 1> output;
    output = call_nn_u(input);
    output = denormalize_u(output);

    for (float e : output) {
        std::cout << e << std::endl;
    }

    
}
