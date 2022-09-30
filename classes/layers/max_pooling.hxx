//
// Created by Malav Patel on 9/29/22.
//

#ifndef CNN_MAX_POOLING_HXX
#define CNN_MAX_POOLING_HXX

class MaxPooling {
    public:

        //! CONSTRUCTORS, DESTRUCTORS, MOVE CONSTRUCTORS, ASSIGNMENT OPERATORS, ETC ------------------------------------

        // create a MaxPool object
        MaxPooling() = default;

        // create MaxPool object from given parameters
        MaxPooling(size_t in_maps, size_t in_width, size_t in_height, size_t fld_width, size_t fld_height, size_t h_stride, size_t v_stride);

        // release allocated memory for MaxPool object
        ~MaxPooling() = default;

        //!------------------------------------------------------------------------------------------------------------

        //! BOOST::APPLY_VISITOR FUNCTIONS ---------------------------------------------------------------------------

        // send feature through the MaxPool layer
        void Forward(std::vector<Vector<double>>& input, std::vector<Vector<double>>& output);

        // send feature backward through the MaxPool layer
        void Backward(std::vector<Vector<double>> &dLdY, std::vector<Vector<double>> &dLdX);

        // get output shape of pooling layer
        Dims const& out_shape() const {return _out;}

        // get input shape of pooling layer
        Dims const& in_shape() const {return _in;}

        // update parameters in this layer (during learning)
        template<typename Optimizer>
        void Update_Params(Optimizer* optimizer, size_t normalizer) {/* do nothing, no parameters to learn in this layer */}

        //! ----------------------------------------------------------------------------------------------------------


        //! OTHER ---------------------------------------------------------------------------------------------------
        std::vector<MaxPool> const & get_pool_vector() const {return pool_vector;}
        //! ---------------------------------------------------------------------------------------------------------
    private:

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

        // vector of individual MaxPool Objects
        std::vector<MaxPool> pool_vector;

};


#include "max_pooling_impl.hxx"
#endif //CNN_MAX_POOLING_HXX
