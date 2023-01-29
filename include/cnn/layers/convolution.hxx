//
// Created by malav on 4/26/2022.
//

#ifndef ANN_CONVOLUTION_HXX
#define ANN_CONVOLUTION_HXX


namespace CNN {

    class Convolution : public Layer {
    public:

        //! CONSTRUCTORS, DESTRUCTORS, MOVE CONSTRUCTORS, ASSIGNMENT OPERATORS, ETC ------------------------------------

        // default constructor should not exist
        Convolution() = delete;

        // create Convolution object from given parameters
        Convolution(size_t in_maps, size_t out_maps, size_t in_width, size_t in_height, size_t filter_width,
                    size_t filter_height, size_t stride_h, size_t stride_v, size_t padleft, size_t padright,
                    size_t padtop,
                    size_t padbottom,
                    double *weights = nullptr);

        // create Convolution object from given parameters
        Convolution(size_t in_maps, size_t out_maps, size_t in_width, size_t in_height, size_t filter_width,
                    size_t filter_height, size_t stride_h, size_t stride_v, bool padding = false,
                    double *weights = nullptr);


        //! -----------------------------------------------------------------------------------------------------------


        //! BOOST::APPLY_VISITOR FUNCTIONS ----------------------------------------------------------------------------

        // send feature through the convolutional layer
        void Forward(Array<double> &input, Array<double> &output) override;

        // send feature backward through convolutional layer, storing gradients in _dLdFs
        void Backward(Array<double> &dLdYs, Array<double> &dLdXs) override;

        // update the weights according to their gradients
        template<typename Optimizer>
        void Update_Params(Optimizer *optimizer, size_t normalizer);

        //! -----------------------------------------------------------------------------------------------------------

        //! OTHER ----------------------------------------------------------------------------------------------------

        // access the filters
        Array<double> const &get_filters() const { return _filters; }

        // return strides
        Dims const get_stride() const { return {_h_str, _v_str}; }

        // return padding
        Dimensions4<> get_padding() const { return {_padleft, _padright, _padtop, _padbottom}; }

        //! -----------------------------------------------------------------------------------------------------------

    private:

        // stores the filters
        Array<double> _filters{};

        // locally stored filter gradients _dLdFs
        Array<double> _dLdFs{};

        // locally stored input feature maps
        Array<double> _local_input{};

        // horizontal stride length
        size_t _h_str{1};

        // vertical stride length
        size_t _v_str{1};

        // padding
        size_t _padleft{0};

        // padding
        size_t _padright{0};

        // padding
        size_t _padtop{0};

        // padding
        size_t _padbottom{0};

    };

}

#include "convolution_impl.hxx"
#endif //ANN_CONVOLUTION_HXX
