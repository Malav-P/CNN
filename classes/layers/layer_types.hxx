//
// Created by malav on 4/26/2022.
//

#ifndef ANN_LAYER_TYPES_HXX
#define ANN_LAYER_TYPES_HXX

#include <boost/variant.hpp>
#include "layer.hxx"

// convolutional layer
class Convolution;

// MaxPooling Layer
class MaxPooling;

// applies mean pooling operation to incoming _data
class MeanPooling;

// a standard y = W*x + b layer with _weights W, bias b
class Linear;

// softmax layer
class Softmax;

// relU layer
class RelU;

// Sigmoid layer
class Sigmoid;

// Tanh layer
class Tanh;


using LayerTypes = boost::variant<
        Convolution*,
        MaxPooling*,
        MeanPooling*,
        Linear*,
        Softmax*,
        RelU*,
        Sigmoid*,
        Tanh*
        >;


#include "convolution.hxx"
#include "max_pooling.hxx"
#include "linear.hxx"
#include "mean_pooling.hxx"
#include "softmax.hxx"
#include "relU.hxx"
#include "sigmoid.hxx"
#include "tanh.hxx"

#endif //ANN_LAYER_TYPES_HXX
