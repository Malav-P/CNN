//
// Created by malav on 4/26/2022.
//

#ifndef ANN_MAX_POOL_HXX
#define ANN_MAX_POOL_HXX

#include "../lin_alg/vector.hxx"

class MaxPool {
    public:

        //! CONSTRUCTORS, DESTRUCTORS, MOVE CONSTRUCTORS, ASSIGNMENT OPERATORS, ETC ------------------------------------

        // create a MaxPool object
        MaxPool() = default;

        // create MaxPool object from given parameters
        MaxPool(size_t in_width, size_t in_height, size_t fld_width, size_t fld_height, size_t h_stride, size_t v_stride);

        // release allocated memory for MaxPool object
        ~MaxPool() = default;

        //!------------------------------------------------------------------------------------------------------------

        //! BOOST::APPLY_VISITOR FUNCTIONS ---------------------------------------------------------------------------

        // send feature through the MaxPool layer
        void Forward(Vector<double>& input, Vector<double>& output);

        // send feature backward through the MaxPool layer
        void Backward(Vector<double>& dLdY, Vector<double>& dLdX);

        // get output shape of pooling layer
        Dims const& out_shape() const {return _out;}

        // get input shape of pooling layer
        Dims const& in_shape() const {return _in;}

        // update parameters in this layer (during learning)
        template<typename Optimizer>
        void Update_Params(Optimizer* optimizer, size_t normalizer) {/* do nothing, no parameters to learn in this layer */}

        //! ----------------------------------------------------------------------------------------------------------


        //! OTHER ---------------------------------------------------------------------------------------------------
        // access the _winners (FOR TESTING PURPOSES)
        Vector<size_t> const& get_winners()  const {return _winners;}
        //! ---------------------------------------------------------------------------------------------------------
    private:

        // helper function for Forward. Returns max value of an array of elements
        Pair max_value(Pair* arr, size_t n);

        // input shape
        Dims _in {0, 0};

        // output shape
        Dims _out {0, 0};

        // field shape
        Dims _field {0, 0};

        // horizontal stride_length
        size_t _h_str {0};

        // vertical stride_length
        size_t _v_str {0};

        // a vector that keeps track of which indices in the the pooling layer are "winning units"
        Vector<size_t> _winners {};

};

#include "max_pool_impl.hxx"
#endif //ANN_MAX_POOL_HXX
