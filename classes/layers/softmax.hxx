//
// Created by malav on 6/27/2022.
//

#ifndef ANN_SOFTMAX_HXX
#define ANN_SOFTMAX_HXX

class Softmax: public Layer {
    public:

        //! CONSTRUCTORS, DESTRUCTORS, MOVE CONSTRUCTORS, ASSIGNMENT OPERATORS, ETC ------------------------------------

        // default constructor shouldnt exist
        Softmax() = delete;

        // constructor
        explicit Softmax(size_t len, double beta = -1)
        :
          Layer(1, len, 1, 1, len, 1),
          _beta(beta),
          _normalization(-1),
          _normalized(false),
          _jacobian(len, len)
        {}
        //! ----------------------------------------------------------------------------------------------------------

        //! BOOST::APPLY_VISITOR FUNCTIONS ---------------------------------------------------------------------------
        // send vector through softmax layer
        void Forward(Vector<double>& input, Vector<double>& output) override
        {
            // check to make sure input and output is of same length as softmax length
            assert(input.get_len() == _in.height && output.get_len() == _in.height);

            // calculate normalization factor
            _normalization = 0;
            for (size_t i = 0; i<_in.height ; i++)
            {
                _normalization += exp(-_beta * input[i]);
            }

            // set normalization flag to true
            _normalized = true;

            // fill in output vector
            for (size_t i = 0; i<_in.height ; i++)
            {
                output[i] = (1/_normalization) * exp(-_beta * input[i]);
            }


            // fill in derivative matrix
            for (size_t i = 0; i < _in.height; i++) { for (size_t j = 0; j < _in.height; j++)
                {
                    _jacobian(i,j) = (-_beta * (exp(-_beta * input[j])/_normalization))*(((i == j) ? 1 : 0 ) - (exp(-_beta * input[i]) / _normalization));
                }}

        }

        // send vector backward through the layer
        void Backward(Vector<double>& dLdY, Vector<double>& dLdX) override
        {
            // check to ensure a forward pass has occurred
            assert(_normalized);
            dLdX = dLdY * _jacobian;

            // reset normalization to -1
            _normalized = false;
        }

        // update parameters for the layer
        template<typename Optimizer>
        void Update_Params(Optimizer* optimizer, size_t normalizer) {/* nothing to do */}

        //! ----------------------------------------------------------------------------------------------------------

        double const& get_beta() const {return _beta;}

    private:

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
