#ifndef FC_LAYER_HPP
#define FC_LAYER_HPP

#include <iostream>
#include <memory>
#include <array>
#include <concepts>


namespace embedded_nn_inference{
using namespace std;


void print_hallo();

template<typename T>
concept nn_callable = requires (T a, T b) {
    a + b; 
    a * b; 
    a = b;
};

template<typename T, typename NNType>
concept activation_function = requires (T v, NNType a) {
    { v.call(a) } -> std::same_as<NNType>;
};



template <nn_callable NNType, typename ActivationFunction, unsigned int width, unsigned int height> 
    requires activation_function<ActivationFunction, NNType>
class FCLayer {
    private:
        const array<array<NNType, width>, height> &weight_data;
        const array<NNType, height> &bias_data;
        ActivationFunction &activation_function;
    
    public:
        FCLayer(const array<array<NNType, width>, height> &weight_data, const array<NNType, height> &bias_data, ActivationFunction &activation_function);
        void print();
        void call(const array<NNType, width> &input, array<NNType, height> &output);
};

template <nn_callable NNType, typename ActivationFunction, unsigned int width, unsigned int height>
requires activation_function<ActivationFunction, NNType>
FCLayer<NNType, ActivationFunction, width, height>::FCLayer(const array<array<NNType, width>, height> &weight_data, const array<NNType, height> &bias_data, ActivationFunction &activation_function) : weight_data(weight_data), bias_data(bias_data), activation_function(activation_function)
{
}
    
template <nn_callable NNType, typename ActivationFunction, unsigned int width, unsigned int height>
requires activation_function<ActivationFunction, NNType>
void FCLayer<NNType, ActivationFunction, width, height>::print()
{
    
    for (size_t i = 0; i < width; i++) {
        for (size_t j = 0; j < height; j++) {
            std::cout << this->weight_data[i][j] << ",";
        }
        std:cout << std::endl;
    }

    for (size_t i = 0; i < height; i++) {
        std::cout << this->bias_data[i] << std::endl;
    }
}

template <nn_callable NNType, typename ActivationFunction, unsigned int width, unsigned int height>
requires activation_function<ActivationFunction, NNType>
void FCLayer<NNType, ActivationFunction, width, height>::call(const array<NNType, width> &input, array<NNType, height> &output)
{
    for (size_t i = 0; i < height; i++) {
        output[i] = this->bias_data[i];
        for (size_t j = 0; j < width; j++) {
            output[i] += this->weight_data[i][j] * input[j];
        }
        output[i] = this->activation_function.call(output[i]);
    }
}

} // namespace embedded_nn_inference

#endif // FC_LAYER_HPP