//
// Created by malav on 4/26/2022.
//

#ifndef ANN_CONVOLUTION_HXX
#define ANN_CONVOLUTION_HXX



namespace CNN {

/**
 * Convolutional layer class
 *
 * ...
 *
 * This class defines the convolutional layer of a network. The layer convolves a kernel with an input image
 * and writes to the given output.
 */

class Convolution : public Layer {
    public:


        /**
         * Default constructor is not callable
         */
        Convolution() = delete;

        /**
         * Constructor for Convolution Class
         *
         * @param in_maps number of input feature maps or input channels to the layer
         * @param out_maps number of output feature maps or output channels from the layer
         * @param in_width input image width
         * @param in_height input image height
         * @param filter_width filter (also called kernel) width
         * @param filter_height filter (also called kernel) height
         * @param stride_h horizontal stride length
         * @param stride_v vertical stride length
         * @param padleft number of columns of zeros to add on left side of image
         * @param padright number of columns of zeros to add on right side of image
         * @param padtop number of rows of zeros to add on top of image
         * @param padbottom number of rows of zeros to add below image
         * @param weights pointer to weights (usually loaded from a file)
         */
        Convolution(size_t in_maps, size_t out_maps, size_t in_width, size_t in_height, size_t filter_width,
                    size_t filter_height, size_t stride_h, size_t stride_v, size_t padleft, size_t padright,
                    size_t padtop,
                    size_t padbottom,
                    double *weights = nullptr);

        /**
         * Alternative constructor for Convolution class
         *
         * This constructor replaces the four unsigned integers padleft, padright, padtop, and padbottom, with a
         * boolean called 'padding', described below.
         *
         * @param in_maps number of input feature maps or input channels to the layer
         * @param out_maps number of output feature maps or output channels from the layer
         * @param in_width input image width
         * @param in_height input image height
         * @param filter_width filter (also called kernel) width
         * @param filter_height filter (also called kernel) height
         * @param stride_h horizontal stride length
         * @param stride_v vertical stride length
         * @param padding set to 'true' for SAME padding or 'false' for VALID padding
         * @param weights pointer to weights (usually loaded from a file)
         */
        Convolution(size_t in_maps, size_t out_maps, size_t in_width, size_t in_height, size_t filter_width,
                    size_t filter_height, size_t stride_h, size_t stride_v, bool padding = false,
                    double *weights = nullptr);


        /**
         * Send image through the layer
         *
         * Convolve the input image with the kernels in this layer. A copy of the input is stored inside the
         * Convolution object for later processing in the 'Backward' member function.
         *
         * @param input an 'Array' object containing the input image.
         * @param output  and 'Array' object containing the output image.
         */
        void Forward(Array<double> &input, Array<double> &output) override;

        /**
         * Backpropagate gradients through this layer.
         *
         * Calculate gradients with respect to the input and the weights. The gradients for the weights are
         * stored locally in the _dLdFs class member.
         * @param dLdYs
         * @param dLdXs
         */
        void Backward(Array<double> &dLdYs, Array<double> &dLdXs) override;

        /**
         * Update the parameters in the layer
         *
         * @tparam Optimizer the optimizer used for updating the weights (adam, rmsprop, sgd, etc.). Optimizers
         * will be defined as classes.
         * @param optimizer pointer to an Optimizer class containing the necessary parameters
         * @param normalizer a constant used to normalize the update, usually equal to the batch size
         */
        template<typename Optimizer>
        void Update_Params(Optimizer *optimizer, size_t normalizer);


        /**
         * Utility function to access the weights (read-only)
         *
         * @return a const reference to the weights for the layer
         */
        Array<double> const &get_filters() const { return _filters; }

        /**
         * Utility function to access the stride information (read-only)
         *
         * @return a const tuple containing the horizontal and vertical strides, respectively
         */
        Dims const get_stride() const { return {_h_str, _v_str}; }

        /**
         * Utility function to get the padding information (read-only)
         *
         * @return a const tuple containing the padding values around the borders of the image
         */
        Dimensions4<> get_padding() const { return {_padleft, _padright, _padtop, _padbottom}; }

        //! -----------------------------------------------------------------------------------------------------------

    private:

        /// locally stored weights
        Array<double> _filters{};

        /// locally stored filter gradients _dLdFs
        Array<double> _dLdFs{};

        /// locally stored input image
        Array<double> _local_input{};

        /// horizontal stride length
        size_t _h_str{1};

        /// vertical stride length
        size_t _v_str{1};

        /// number of cols of padding to the left of image
        size_t _padleft{0};

        /// number of cols of padding to the right of image
        size_t _padright{0};

        /// number of cols of padding above the image
        size_t _padtop{0};

        /// number of cols of padding below the image
        size_t _padbottom{0};

    };

}

#include "convolution_impl.hxx"
#endif //ANN_CONVOLUTION_HXX
