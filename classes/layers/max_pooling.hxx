//
// Created by Malav Patel on 9/29/22.
//

#ifndef CNN_MAX_POOLING_HXX
#define CNN_MAX_POOLING_HXX

class MaxPooling {
    public:

        //! CONSTRUCTORS, DESTRUCTORS, MOVE CONSTRUCTORS, ASSIGNMENT OPERATORS, ETC ------------------------------------

        // create MaxPool object from given parameters
        MaxPooling(size_t in_maps, size_t in_width, size_t in_height, size_t fld_width, size_t fld_height, size_t h_stride, size_t v_stride);

        // release allocated memory for MaxPool object
        ~MaxPooling()
        {
            for(size_t i = 0 ; i < _in_maps ; i++)
            {
                //free device memory
                cudaFree(d_winners[i]);
            }

            // free memory
            free(d_winners);

            // free device memory
            cudaFree(d_poolvec);

            delete[] pool_vector;
        }

        //!------------------------------------------------------------------------------------------------------------

        //! BOOST::APPLY_VISITOR FUNCTIONS ---------------------------------------------------------------------------

        // send feature through the MaxPool layer
        __host__ void Forward(Vector<double> &input, Vector<double> &output);

        // send feature backward through the MaxPool layer
        void Backward(Vector<double> &dLdY, Vector<double> &dLdX);

        // get output shape of pooling layer
        Dims3 const& out_shape() const {return _out;}

        // get input shape of pooling layer
        Dims3 const& in_shape() const {return _in;}

        // update parameters in this layer (during learning)
        template<typename Optimizer>
        void Update_Params(Optimizer* optimizer, size_t normalizer) {/* do nothing, no parameters to learn in this layer */}

        //! ----------------------------------------------------------------------------------------------------------


        //! OTHER ---------------------------------------------------------------------------------------------------
        MaxPool*  get_pool_vector() const {return pool_vector;}
        //! ---------------------------------------------------------------------------------------------------------
    private:

        // input shape
        Dims3 _in {0, 0,0};

        // output shape
        Dims3 _out {0, 0,0};

        // field shape
        Dims _field {0, 0};

        // horizontal stride_length
        size_t _h_str {0};

        // vertical stride_length
        size_t _v_str {0};

        // vector of individual MaxPool Objects
        MaxPool* pool_vector {nullptr};

        // device version of MaxPool Objects
        MaxPool* d_poolvec {nullptr};

        // device version of _winners attribute of MaxPool class
        size_t ** d_winners {nullptr};

        // size of pool_vector
        size_t _in_maps {0};

};


#include "max_pooling_impl.hxx"
#endif //CNN_MAX_POOLING_HXX
