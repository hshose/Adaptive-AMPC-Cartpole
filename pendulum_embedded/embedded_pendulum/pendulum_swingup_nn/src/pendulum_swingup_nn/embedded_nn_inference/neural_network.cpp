#include <array>
#include "neural_network.hpp"

namespace embedded_nn_inference{

    using namespace std;

    

    namespace u {
    
    array<float, 1>
    call_nn(const array<float, 4>& input){
        
        layer1.call(input, layer_output1);
        
        layer2.call(layer_output1, layer_output2);
        
        layer3.call(layer_output2, layer_output3);
        
        layer4.call(layer_output3, layer_output4);
        
        layer5.call(layer_output4, layer_output5);
        return layer_output5;
    }

    array<float, 4>
    normalize_input(const array<float, 4>& in){
        array<float, 4> res;
        for (int i = 0; i < 4; i++) {
            res[i] = (in[i] - input_offset[i]) / input_scale[i];
        }
        return res;
    }
    
    array<float, 1>
    denormalize_output(const array<float, 1>& out){
        array<float, 1> res;
        for (int i = 0; i < 1; i++) {
            res[i] = out[i]*output_scale[i] + output_offset[i];
        }
        return res;
    }

    } // namespace u 
    

    namespace J {
    
    array<float, 5>
    call_nn(const array<float, 4>& input){
        
        layer1.call(input, layer_output1);
        
        layer2.call(layer_output1, layer_output2);
        
        layer3.call(layer_output2, layer_output3);
        
        layer4.call(layer_output3, layer_output4);
        
        layer5.call(layer_output4, layer_output5);
        
        layer6.call(layer_output5, layer_output6);
        
        layer7.call(layer_output6, layer_output7);
        
        layer8.call(layer_output7, layer_output8);
        return layer_output8;
    }

    array<float, 4>
    normalize_input(const array<float, 4>& in){
        array<float, 4> res;
        for (int i = 0; i < 4; i++) {
            res[i] = (in[i] - input_offset[i]) / input_scale[i];
        }
        return res;
    }
    
    array<float, 5>
    denormalize_output(const array<float, 5>& out){
        array<float, 5> res;
        for (int i = 0; i < 5; i++) {
            res[i] = out[i]*output_scale[i] + output_offset[i];
        }
        return res;
    }

    } // namespace J 
    

}
    