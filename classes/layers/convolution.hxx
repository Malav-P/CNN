//
// Created by malav on 4/26/2022.
//

#ifndef ANN_CONVOLUTION_HXX
#define ANN_CONVOLUTION_HXX

#include "../lin_alg/data_types.hxx"

class Convolution {
    public:

        //! CONSTRUCTORS, DESTRUCTORS, MOVE CONSTRUCTORS, ASSIGNMENT OPERATORS, ETC ------------------------------------

        //create a Convolution object
        Convolution() = default;

        // create Convolution object from given parameters
        Convolution(size_t in_width, size_t in_height, const Mat<double>& filter, size_t stride_h , size_t stride_v , size_t padleft,
                    size_t padright, size_t padtop, size_t padbottom);

        // create Convolution object from given parameters
        Convolution(size_t in_width, size_t in_height, const Mat<double>& filter, size_t stride_h , size_t stride_v , bool padding = false);

        // release allocated memory for Convolution object
        ~Convolution() {delete[] _indices;}

        //! -----------------------------------------------------------------------------------------------------------

        //! BOOST::APPLY_VISITOR FUNCTIONS ----------------------------------------------------------------------------

        // send feature through the convolutional layer
        void Forward(Vector<double>& input, Vector<double>& output);

        // send feature backward through convolutional layer, keeping track of gradients
        void Backward(Vector<double>& dLdY, Vector<double>& dLdX);

        // get output shape of convolution
        Dims const& out_shape() const {return _out;}

        // get input shape of convolution
        Dims const& in_shape() const {return _in;}

        // update the weights and biases according to their gradients
        template<typename Optimizer>
        void Update_Params(Optimizer* optimizer, size_t normalizer);

        //! -----------------------------------------------------------------------------------------------------------

        //! OTHER ----------------------------------------------------------------------------------------------------

        // access the kernel (FOR TESTING PURPOSES)
        Mat<double> const& get_kernel()  const {return _kernel;}

        // access the indices
        Dims* const& get_indices() const {return _indices;}
        //! -----------------------------------------------------------------------------------------------------------

    private:

        // stores the _data for the kernel as toeplitz matrix
        Mat<double> _kernel {};

        // keep track of important indices in kernel
        Dims* _indices {nullptr};

        // locally stored gradients dL/dW
        Mat<double> _dLdW {};

        // locally stored input
        Vector<double> _local_input {};

        // horizontal stride length
        size_t _h_str {1};

        // vertical stride length
        size_t _v_str {1};

        // input image shape
        Dims _in {0, 0};

        // output image shape
        Dims _out {0, 0};

        // padding
        size_t _padleft {0};

        // padding
        size_t _padright {0};

        // padding
        size_t _padtop {0};

        // padding
        size_t _padbottom {0};


};


#include "convolution_impl.hxx"
#endif //ANN_CONVOLUTION_HXX
