//
// Created by malav on 4/26/2022.
//

#ifndef ANN_MEAN_POOL_HXX
#define ANN_MEAN_POOL_HXX


class MeanPool {
public:

    //! CONSTRUCTORS, DESTRUCTORS, MOVE CONSTRUCTORS, ASSIGNMENT OPERATORS, ETC ------------------------------------

    // default constructor shouldnt exist
    MeanPool() = default;

    // create MaxPool object from given parameters
    MeanPool(size_t in_width, size_t in_height, size_t fld_width, size_t fld_height, size_t h_stride, size_t v_stride);


    //! --------------------------------------------------------------------------------------------------------------

    //! BOOST::APPLY_VISITOR FUNCTIONS ------------------------------------------------------------------------------

    // send feature through the MeanPool layer
    void Forward(Vector<double>& input, Vector<double>& output);

    // send feature backward through the MeanPool Layer
    void Backward(Vector<double>& dLdY, Vector<double>& dLdX);

    // get output shape
    Dims3 const& out_shape() const {return _out;}

    // get input shape
    Dims3 const& in_shape() const {return _in;}

    // update parameters for the layer (during learning)
    template<typename Optimizer>
    void Update_Params(Optimizer* optimizer, size_t normalizer) {/* do nothing, no parameters to learn in this layer */}

    //! ---------------------------------------------------------------------------------------------------------------

private:

    // helper function for Forward. Returns max value of an array of elements
    template<typename T>
    double avg_value(T* arr, size_t n);

    // input shape
    Dims3 _in {0,0,1};

    // output shape
    Dims3 _out {0,0,1};

    // field shape
    Dims _field {0,0};

    // horizontal stride_length
    size_t _h_str {0};

    // vertical stride_length
    size_t _v_str {0};

};

#include "mean_pool_impl.hxx"
#endif //ANN_MEAN_POOL_HXX
