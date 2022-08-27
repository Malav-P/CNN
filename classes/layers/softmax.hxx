//
// Created by malav on 6/27/2022.
//

#ifndef ANN_SOFTMAX_HXX
#define ANN_SOFTMAX_HXX

#include "../prereqs.hxx"
#include "../lin_alg/data_types.hxx"

class Softmax {
    public:

        //! CONSTRUCTORS, DESTRUCTORS, MOVE CONSTRUCTORS, ASSIGNMENT OPERATORS, ETC ------------------------------------

        // default constructor
        Softmax() = default;

        // constructor
        explicit Softmax(size_t len, double beta = -1)
        : _size(len),
          _beta(beta),
          _normalization(-1),
          _normalized(false),
          _jacobian(len, len)
        {}
        //! ----------------------------------------------------------------------------------------------------------

        //! BOOST::APPLY_VISITOR FUNCTIONS ---------------------------------------------------------------------------
        // send vector through softmax layer
        void Forward(Vector<double>& input, Vector<double>& output, bool training = true)
        {
            // check to make sure input and output is of same length as softmax length
            assert(input.get_len() == _size && output.get_len() == _size);

            // calculate normalization factor
            _normalization = 0;
            for (size_t i = 0; i<_size ; i++)
            {
                _normalization += exp(-_beta * input[i]);
            }

            // set normalization flag to true
            _normalized = true;

            // fill in output vector
            for (size_t i = 0; i<_size ; i++)
            {
                output[i] = (1/_normalization) * exp(-_beta * input[i]);
            }

            if (training)
            {
                // fill in derivative matrix
                for (size_t i = 0; i < _size; i++) { for (size_t j = 0; j < _size; j++)
                    {
                        _jacobian(i,j) = (-_beta * (exp(-_beta * input[j])/_normalization))*(((i == j) ? 1 : 0 ) - (exp(-_beta * input[i]) / _normalization));
                    }}
            }
        }

        // send vector backward through the layer
        void Backward(Vector<double>& dLdY, Vector<double>& dLdX)
        {
            // check to ensure a forward pass has occurred
            assert(_normalized);
            dLdX = dLdY * _jacobian;

            // reset normalization to -1
            _normalized = false;
        }

        // get input shape
        Dims in_shape() const {return {1, _size};}

        // get output shape
        Dims out_shape() const {return in_shape();}

        // update parameters for the layer
        template<typename Optimizer>
        void Update_Params(Optimizer* optimizer, size_t normalizer) {/* nothing to do */}

        //! ----------------------------------------------------------------------------------------------------------

    private:

        // size of softmax layer
        size_t _size {0};

        // temperature parameter
        double _beta {-1};

        // normalization factor
        double _normalization {-1};

        // normalization flag
        bool _normalized {false};

        // derivative matrix
        Mat<double> _jacobian {};
};


#endif //ANN_SOFTMAX_HXX
