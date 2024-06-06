#ifndef ACTIVATION_FUNCTION_CPP
#define ACTIVATION_FUNCTION_CPP

#include "activation_function.hpp"
#include "math.h"
#include <iostream>

namespace embedded_nn_inference{

float TANH::call(float a)
{
    int index = (int) ((a - this->min_value) / this->step_size);
    
    if (index <= 0) {
        return -1.0f;
    } 
    if (index >= tanh_num_elem - 1) {
        return 1.0f;
    }
    return (this->tanh_lookup.at(index+1) - this->tanh_lookup.at(index)) / this->step_size * (a - (index * this->step_size + this->min_value)) + this->tanh_lookup.at(index);
}

float LINEAR::call(float a)
{
    return a;
}

} // namespace embedded_nn_inference

#endif // ACTIVATION_FUNCTION_CPP