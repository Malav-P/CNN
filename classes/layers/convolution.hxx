//
// Created by malav on 4/26/2022.
//

#ifndef ANN_CONVOLUTION_HXX
#define ANN_CONVOLUTION_HXX



class Convolution {
    public:

        //! CONSTRUCTORS, DESTRUCTORS, MOVE CONSTRUCTORS, ASSIGNMENT OPERATORS, ETC ------------------------------------

        //create a Convolution object
        Convolution() = default;

        // create Convolution object from given parameters
        Convolution(size_t in_maps, size_t out_maps, size_t in_width, size_t in_height, size_t filter_width,
                    size_t filter_height, size_t stride_h, size_t stride_v, size_t padleft, size_t padright,
                    size_t padtop,
                    size_t padbottom);

        // create Convolution object from given parameters
        Convolution(size_t in_maps, size_t out_maps, size_t in_width, size_t in_height, size_t filter_width,
                    size_t filter_height, size_t stride_h, size_t stride_v, bool padding = false);

        // release allocated memory for Convolution object
        ~Convolution()
        {
            for(size_t i = 0 ; i < _out.depth ; i++)
            {
                //free device memory for device data arrays
                cudaFree(d_filters_data[i]);
                cudaFree(d_dLdFs_data[i]);
            }

            // free host data arrays
            delete[] d_dLdFs_data;
            delete[] d_filters_data;

            // free device struct arrays
            cudaFree(d_filters);
            cudaFree(d_dLdFs);

            delete[] _filters;
            delete[] _dLdFs;
            delete[] _local_input;
        }

        //! -----------------------------------------------------------------------------------------------------------

        //! BOOST::APPLY_VISITOR FUNCTIONS ----------------------------------------------------------------------------

        // send feature through the convolutional layer
        void Forward(Vector<double> &input, Vector<double> &output);

        // send feature backward through convolutional layer, keeping track of gradients
        void Backward(Vector<double> &dLdYs, Vector<double> &dLdXs);

        // get output shape of convolution
        Dims3 const& out_shape() const {return _out;}

        // get input shape of convolution (without padding!)
        Dims3  in_shape() const  {return {_in.width - _padleft - _padright, _in.height - _padtop - _padbottom, _in.depth};}

        // update the weights and biases according to their gradients
        template<typename Optimizer>
        void Update_Params(Optimizer* optimizer, size_t normalizer);

        //! -----------------------------------------------------------------------------------------------------------

        //! OTHER ----------------------------------------------------------------------------------------------------

        // access the kernel (FOR TESTING PURPOSES)
        Cuboid<double>* const& get_filters()  const {return _filters;}

        // access the local input
        Mat<double>* const& get_local_input() const {return _local_input;}

        // print the filters
        void print_filters();

        //! -----------------------------------------------------------------------------------------------------------

    public:

        // horizontal stride length
        size_t _h_str {1};

        // vertical stride length
        size_t _v_str {1};

        // input image shape
        Dims3 _in ;

        // output image shape
        Dims3 _out {0, 0,0};

        // padding
        size_t _padleft {0};

        // padding
        size_t _padright {0};

        // padding
        size_t _padtop {0};

        // padding
        size_t _padbottom {0};

        // stores the filter
        Cuboid<double>* _filters {nullptr};

        // device version of Cuboid<double> Objects
        Cuboid<double>* d_filters {nullptr};

        // device version of _data attribute of Cuboid<double> class
        double ** d_filters_data {nullptr};

        // locally stored filter gradient _dLdF
        Cuboid<double>* _dLdFs {nullptr};

        // device version of Cuboid<double> Objects
        Cuboid<double>* d_dLdFs {nullptr};

        // device version of _data attribute of Cuboid<double> class
        double** d_dLdFs_data {nullptr};

        // locally stored input feature maps
        Mat<double>* _local_input {nullptr};

};


#include "convolution_impl.hxx"
#endif //ANN_CONVOLUTION_HXX
