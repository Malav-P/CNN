//
// Created by malav on 4/26/2022.
//

#ifndef ANN_LAYER_TYPES_HXX
#define ANN_LAYER_TYPES_HXX

#include "../activation functions/activation_types.hxx"
#include <boost/variant.hpp>

// convolutional layer
class Convolution;

// applies max pooling operation to incoming _data
class MaxPool;

// applies mean pooling operation to incoming _data
class MeanPool;

// a standard y = f(w*x + b) layer with _weights w, bias b and activation function f
template<typename activ_func>
class Linear;

// softmax layer
class Softmax;

using LayerTypes = boost::variant<
        Convolution*,
        MaxPool*,
        MeanPool*,
        Linear<RelU>*,
        Linear<Sigmoid>*,
        Linear<Tanh>*,
        Softmax*
        >;

#include "convolution.hxx"
#include "linear.hxx"
#include "max_pool.hxx"
#include "mean_pool.hxx"
#include "softmax.hxx"
#endif //ANN_LAYER_TYPES_HXX
