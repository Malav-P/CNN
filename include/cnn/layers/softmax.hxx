//
// Created by malav on 6/27/2022.
//

#ifndef ANN_SOFTMAX_HXX
#define ANN_SOFTMAX_HXX

namespace CNN{

    /**
     * An implementation of the softmax activation, used to output probabilities for object classification
     *
     * @see [Softmax Activation](https://en.wikipedia.org/wiki/Softmax_function)
     */
class Softmax: public Layer {
    public:

        /**
         * Default constructor should not exist, only defined constructor can be called
         */
        Softmax() = delete;

        /**
         * Constructor for Softmax class
         *
         * @param len number of output classification categories
         * @param beta the temperature parameter, defaults to -1
         */
        explicit Softmax(size_t len, double beta = -1)
        :
          Layer(1, len, 1, 1, len, 1),
          _beta(beta),
          _normalization(-1),
          _normalized(false),
          _jacobian({len, len})
        {}

        /**
         * Propagate data forward through this layer
         * @param input input image
         * @param output output where results of Softmax activation on input will be stored
         */
        void Forward(Array<double>& input, Array<double>& output) override
        {
            // check to make sure input and output is of same length as softmax length
            assert(input.getsize() == _in.height && output.getsize() == _in.height);

            // calculate normalization factor
            _normalization = 0;
            for (size_t i = 0; i<_in.height ; i++)
            {
                _normalization += exp(-_beta * input[{0,i}]);
            }

            // set normalization flag to true
            _normalized = true;

            // fill in output vector
            for (size_t i = 0; i<_in.height ; i++)
            {
                output[{0,i}] = (1/_normalization) * exp(-_beta * input[{0,i}]);
            }


            // fill in derivative matrix
            for (size_t i = 0; i < _in.height; i++) { for (size_t j = 0; j < _in.height; j++)
                {
                    _jacobian[{i,j}] = (-_beta * (exp(-_beta * input[{0,j}])/_normalization))*(((i == j) ? 1 : 0 ) - (exp(-_beta * input[{0,i}]) / _normalization));
                }}

        }

        /**
         * Propagate gradients backwards through layer and compute loss gradient wrt input
         *
         * @param dLdY loss gradient wrt output
         * @param dLdX where the loss gradient wrt input will be stored
         */
        void Backward(Array<double>& dLdY, Array<double>& dLdX) override
        {
            // check to ensure a forward pass has occurred
            assert(_normalized);

            // num rows in C
            int l = dLdX.getshape()[0];
            // num cols in C
            int n = dLdX.getshape()[1];
            // num cols in A
            int m = dLdY.getshape()[1];
            double alpha = 1.0;
            int lda = dLdY.getshape()[1];
            int ldb = _jacobian.getshape()[1];
            double beta = 0.0;
            int ldc = dLdX.getshape()[1];

            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, l, n,m,alpha, dLdY.getdata(), lda, _jacobian.getdata(), ldb, beta, dLdX.getdata(), ldc);

            //dLdX = dLdY * _jacobian;

            // reset normalization to -1
            _normalized = false;
        }

        /**
         * Update the parameters in the layer
         *
         * @tparam Optimizer the optimizer used for updating the weights (adam, rmsprop, sgd, etc.). Optimizers
         * will be defined as classes.
         * @param optimizer pointer to an Optimizer class containing the necessary parameters
         * @param normalizer a constant used to normalize the update, usually equal to the batch size
         */
        template<typename Optimizer>
        void Update_Params(Optimizer* optimizer, size_t normalizer) {/* nothing to do */}

        /**
         * Utility function to access the temperature parameter (read-only)
         *
         * @return a constant reference to the temperature, _beta
         */
        double const& get_beta() const {return _beta;}

    private:

        /// temperature parameter
        double _beta {-1};

        /// normalization factor
        double _normalization {-1};

        /// normalization flag
        bool _normalized {false};

        /// derivative matrix
        Array<double> _jacobian {};
};

}
#endif //ANN_SOFTMAX_HXX
