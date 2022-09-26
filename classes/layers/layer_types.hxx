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

// a standard y = W*x + b layer with _weights W, bias b
class Linear;

// softmax layer
class Softmax;

// relU layer
class RelU;

using LayerTypes = boost::variant<
        Convolution*,
        MaxPool*,
        MeanPool*,
        Linear*,
        Softmax*,
        RelU*
        >;


#include "../lin_alg/data_types.hxx"

#include "convolution.hxx"
#include "linear.hxx"
#include "max_pool.hxx"
#include "mean_pool.hxx"
#include "softmax.hxx"
#include "relU.hxx"
#endif //ANN_LAYER_TYPES_HXX
