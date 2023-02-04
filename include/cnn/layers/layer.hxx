//
// Created by Malav Patel on 11/7/22.
//

#ifndef CNN_LAYER_HXX
#define CNN_LAYER_HXX

#include "../lin_alg/data_types.hxx"

namespace CNN {

/**
 * The base class for layer in the convolutional neural network. All layers inherit from this class.
 */
class Layer {
    public:

        /**
         * Default constructor should never be called
         */
        Layer() = delete;

        /**
         * Constructor for the 'Layer' class.
         * @param in_width  width of incoming 'Array' object
         * @param in_height  height of incoming 'Array' object
         * @param in_depth  depth of incoming 'Array' object
         * @param out_width width of outgoing 'Array' object
         * @param out_height height of outgoing 'Array' object
         * @param out_depth depth of outgoing 'Array' object
         */
        Layer(size_t in_width, size_t in_height, size_t in_depth, size_t out_width, size_t out_height, size_t out_depth)
                : _in(in_width, in_height, in_depth),
                  _out(out_width, out_height, out_depth) {}


        /**
         * Propagate data forward through the layer
         *
         *
         * @param input the input data as an 'Array' object
         * @param output where to write the output data as an 'Array' object
         *
         * @note This member function will always be overridden by a class that inherits it
         */
        virtual void Forward(Array<double> &input, Array<double> &output) = 0;

        /**
         * Propagate gradients backwards through the layer
         *
         * @param dLdYs
         * @param dLdXs
         *
         * @note This member function will always be overridden by a class that inherits it
         */
        virtual void Backward(Array<double> &dLdYs, Array<double> &dLdXs) = 0;

        /**
         * Utility function to get the output shape of this layer
         *
         * @return a const reference to a 3-tuple of the output dimensions of the layer
         */
        Dims3 const &out_shape() const { return _out; }

        /**
         * Utility function to get the input shape of this layer
         *
         * @return a const reference to a 3-tuple of the input dimensions of the layer
         *
         * @note the dimensions returned are those without padding!
         */
        Dims3 in_shape() const { return _in; }

    protected:

        /// input shape of layer
        Dims3 _in{0, 0, 0};

        /// output shape of layer
        Dims3 _out{0, 0, 0};

    };

}
#endif //CNN_LAYER_HXX
