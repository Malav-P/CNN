//
// Created by Malav Patel on 9/29/22.
//

#ifndef CNN_MAX_POOLING_HXX
#define CNN_MAX_POOLING_HXX


class MaxPool {
public:

    //! CONSTRUCTORS, DESTRUCTORS, MOVE CONSTRUCTORS, ASSIGNMENT OPERATORS, ETC ------------------------------------

    // default constructor
    MaxPool() = default;

    // create MaxPool object from given parameters
    MaxPool(size_t in_width, size_t in_height, size_t fld_width, size_t fld_height, size_t h_stride, size_t v_stride);


    //!------------------------------------------------------------------------------------------------------------

    //! BOOST::APPLY_VISITOR FUNCTIONS ---------------------------------------------------------------------------

    // send feature through the MaxPool layer
    void Forward(Vector<double>& input, Vector<double>& output);

    // send feature backward through the MaxPool layer
    void Backward(Vector<double>& dLdY, Vector<double>& dLdX);

    // get output shape of pooling layer
    Dims3 const& out_shape() const {return _out;}

    // get input shape of pooling layer
    Dims3 const& in_shape() const {return _in;}

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
    Dims3 _in {0, 0,1};

    // output shape
    Dims3 _out {0, 0,1};

    // field shape
    Dims _field {0, 0};

    // horizontal stride_length
    size_t _h_str {0};

    // vertical stride_length
    size_t _v_str {0};

    // a vector that keeps track of which indices in the pooling layer are "winning units"
    Vector<size_t> _winners {};

};

class MaxPooling: public Layer {
    public:

        //! CONSTRUCTORS, DESTRUCTORS, MOVE CONSTRUCTORS, ASSIGNMENT OPERATORS, ETC ------------------------------------

        // default constructor shouldn't exist
        MaxPooling() = delete;

        // create MaxPool object from given parameters
        MaxPooling(size_t in_maps, size_t in_width, size_t in_height, size_t fld_width, size_t fld_height, size_t h_stride, size_t v_stride);

        //!------------------------------------------------------------------------------------------------------------

        //! BOOST::APPLY_VISITOR FUNCTIONS ---------------------------------------------------------------------------

        // send feature through the MaxPool layer
        void Forward(Vector<double> &input, Vector<double> &output) override;

        // send feature backward through the MaxPool layer
        void Backward(Vector<double> &dLdY, Vector<double> &dLdX) override;

        // update parameters in this layer (during learning)
        template<typename Optimizer>
        void Update_Params(Optimizer* optimizer, size_t normalizer) {/* do nothing, no parameters to learn in this layer */}

        //! ----------------------------------------------------------------------------------------------------------


        //! OTHER ---------------------------------------------------------------------------------------------------
        std::vector<MaxPool> const & get_pool_vector() const {return pool_vector;}

        Dims const& get_field() const {return _field;}

        Dims get_stride() const {return {_h_str, _v_str};}
        //! ---------------------------------------------------------------------------------------------------------
    private:

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
