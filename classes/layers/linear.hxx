//
// Created by malav on 5/2/2022.
//

#ifndef ANN_LINEAR_HXX
#define ANN_LINEAR_HXX


//! a linear class which applies a transformation of the form
//! Y = Wx + b. W is a weight matrix and b is a bias.
class Linear {
    public:

        //! CONSTRUCTORS, DESTRUCTORS, MOVE CONSTRUCTORS, ASSIGNMENT OPERATORS, ETC ------------------------------------

        // default constructor
        Linear() = default;

        // construct Linear Layer with specified input and output sizes
        Linear(size_t in_size, size_t out_size);

        // destructor
        ~Linear()
        {
            // free device memory
            cudaFree(d_weights);
            cudaFree(d_weight_data);

            cudaFree(d_biases);
            cudaFree(d_biases_data);

            cudaFree(d_dLdW);
            cudaFree(d_dLdW_data);

            cudaFree(d_dLdB);
            cudaFree(d_dLdB_data);
        }

        //! -----------------------------------------------------------------------------------------------------------

        //! BOOST::APPLY_VISITOR FUNCTIONS ---------------------------------------------------------------------------

        // send vector forward through this layer
        void Forward(Vector<double>& input, Vector<double>& output);

        // send vector backwards through layer, computing gradients and input error dLdX
        void Backward(Vector<double>& dLdY, Vector<double>& dLdX);

        // update the weights and biases according to their gradients
        template<typename Optimizer>
        void Update_Params(Optimizer* optimizer, size_t normalizer);

        //! ----------------------------------------------------------------------------------------------------------

        //! OTHER ----------------------------------------------------------------------------------------------------

        // get the weight matrix
        Mat<double> const& get_weights() const {
            // retrieve data from device and put it into return variable
            cudaMemcpy(_weights._data, d_weight_data, _weights.get_cols()*_weights.get_rows()*sizeof(double), cudaMemcpyDeviceToHost);
            return _weights;
        }

        // get gradient matrix
        Mat<double> const& get_dLdW() const {
            // retrieve data from device and put it into return variable
            cudaMemcpy(_dLdW._data, d_dLdW_data, _weights.get_cols()*_weights.get_rows()*sizeof(double), cudaMemcpyDeviceToHost);
            return _dLdW;}

        // get the biases
        Vector<double> const& get_biases() const {
            // retrieve data from device and put it into return variable
            cudaMemcpy(_biases.get_data(), d_biases, _biases.get_len()*sizeof(double), cudaMemcpyDeviceToHost);
            return _biases;}

        // get the dLdB vector
        Vector<double> const& get_dLdB() const {
        // retrieve data from device and put it into return variable
        cudaMemcpy(_dLdB.get_data(), d_dLdB_data, _biases.get_len()*sizeof(double), cudaMemcpyDeviceToHost);
        return _dLdB;}

        // get local output
        Vector<double> const& get_local_output() const {return _local_output;}

        // return out shape of layer
        Dims3 const& out_shape() const {return _out;}

        // return in shape of layer
        Dims3 const& in_shape() const {return _in;}

        //! ---------------------------------------------------------------------------------------------------------
    private:
        // NOTE: empty braces call default constructor for that class ( at least i hope it does)

        // input shape
        Dims3 _in {0, 0,1};

        // output shape
        Dims3 _out {0,0,1};

        // locally stored input
        Vector<double> _local_input {};

        // locally stored output Y = Wx + B
        Vector<double> _local_output {};

        // weight matrix W
        Mat<double> _weights {};

        // weight matrix on device
        Mat<double>* d_weights {nullptr};

        // weight matrix data on device
        double* d_weight_data {nullptr};

        // bias vector
        Vector<double> _biases {};

        // bias vector on the device
        Vector<double>* d_biases {nullptr};

        // bias vector data on device
        double* d_biases_data {nullptr};

        // locally stored gradients dL/dW
        Mat<double> _dLdW {};

        // gradients matrix on device
        Mat<double>* d_dLdW {nullptr};

        // gradient matrix data on device
        double* d_dLdW_data {nullptr};

        // locally stored gradients dL/dB
        Vector<double> _dLdB {};

        // bias gradient on the device
        Vector<double>* d_dLdB {nullptr};

        // bias gradient data on device
        double* d_dLdB_data {nullptr};

};


#include "linear_impl.hxx"
#endif //ANN_LINEAR_HXX
