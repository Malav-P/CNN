//
// Created by malav on 4/26/2022.
//

#ifndef ANN_MAX_POOL_HXX
#define ANN_MAX_POOL_HXX


class MaxPool {
    public:

        //! CONSTRUCTORS, DESTRUCTORS, MOVE CONSTRUCTORS, ASSIGNMENT OPERATORS, ETC ------------------------------------

        // create a MaxPool object
        MaxPool() = default;

        // create MaxPool object from given parameters
        MaxPool(size_t in_width, size_t in_height, size_t fld_width, size_t fld_height, size_t h_stride, size_t v_stride);

        // move assignment operator
        MaxPool& operator=(MaxPool&& other) noexcept;

        // move constructor
        MaxPool(MaxPool&& other) noexcept;

        // release allocated memory for MaxPool object
        ~MaxPool() {delete[] _winners; }

        //!------------------------------------------------------------------------------------------------------------

        //! BOOST::APPLY_VISITOR FUNCTIONS ---------------------------------------------------------------------------

        // send feature through the MaxPool layer
        __host__ __device__ void Forward(Vector<double>& input, Vector<double>& output);

        // send feature backward through the MaxPool layer
        void Backward(Vector<double>& dLdY, Vector<double>& dLdX);

        // get output shape of pooling layer
        __host__ __device__ Dims3 const& out_shape() const {return _out;}

        // get input shape of pooling layer
        __host__ __device__ Dims3 const& in_shape() const {return _in;}

        // update parameters in this layer (during learning)
        template<typename Optimizer>
        void Update_Params(Optimizer* optimizer, size_t normalizer) {/* do nothing, no parameters to learn in this layer */}

        //! ----------------------------------------------------------------------------------------------------------


        //! OTHER ---------------------------------------------------------------------------------------------------
        // access the _winners (FOR TESTING PURPOSES)
        __host__ __device__ size_t* & get_winners()   {return _winners;}
        //! ---------------------------------------------------------------------------------------------------------
    private:

        friend class MaxPooling;

        // helper function for Forward. Returns max value of an array of elements
        __host__ __device__ Pair max_value(Pair* arr, size_t n);

    public:
        // a vector that keeps track of which indices in the pooling layer are "winning units"
        size_t* _winners {nullptr};

        // input shape
        Dims3 _in{0, 0,1};

        // output shape
        Dims3 _out{0, 0,1};

        // field shape
        Dims _field{0, 0};

        // horizontal stride_length
        size_t _h_str{0};

        // vertical stride_length
        size_t _v_str {0};




};

#include "max_pool_impl.hxx"
#endif //ANN_MAX_POOL_HXX
