//
// Created by malav on 5/4/2022.
//

#ifndef ANN_LINEAR_CPP
#define ANN_LINEAR_CPP

#include "linear.hxx"

namespace CNN {


    Linear::Linear(size_t in_size, size_t out_size, double *weights, double* biases)
            : Layer(1, in_size, 1, 1, out_size, 1), _weights({out_size, in_size}),
              _dLdW({out_size, in_size}), _biases({out_size, 1}), _dLdB({out_size,1}) {

        // get the current time to seed the random number generator
        typedef std::chrono::high_resolution_clock myclock;
        myclock::time_point beginning = myclock::now();
        myclock::duration d = myclock::now() - beginning;
        unsigned seed2 = d.count();

        // seed the random number generator
        std::default_random_engine generator(seed2);
        std::normal_distribution<double> distribution(0, sqrt(2.0 / _in.height));

        // He initialize the weights if no weights are provided,
        if (weights == nullptr) {
            double* data = _weights.getdata();
            for (size_t i = 0; i < _weights.getsize(); i++) {
                data[i] = distribution(generator);
            }
        } else {
            std::memcpy(_weights.getdata(), weights, _weights.getsize() * sizeof(double));
        }

        if (biases != nullptr)
        {
            std::memcpy(_biases.getdata(), biases, out_size*sizeof(double ));
        }

    }


    void Linear::Forward(Array<double> &input, Array<double> &output) {

        // TODO - ensure that output has correct dimensions for matrix multiplication
        // Reshape output to column vector
        output.Reshape({output.getshape()[1], 1});
        assert(output.getshape()[0] == _weights.getshape()[0]);

        // copy input to local variable
        _local_input = input;
        _local_input.Reshape({input.getsize(),1});

        // perform Y = Wx + B
        // combination of dgemm and saxpy


        // num rows in C
        int l = output.getshape()[0];
        // num cols in C
        int n = output.getshape()[1];
        // num cols in A
        int m = _weights.getshape()[1];
        double alpha = 1.0;
        int lda = _weights.getshape()[1];
        int ldb = _local_input.getshape()[1];
        double beta = 0.0;
        int ldc = output.getshape()[1];

        // do matmul
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, l, n,m,alpha, _weights.getdata(), lda, _local_input.getdata(), ldb, beta, output.getdata(), ldc);

        // add biases
        int len = output.getsize();
        double scalar = 1.0;
        cblas_daxpy(len, scalar, _biases.getdata(), 1, output.getdata(), 1);

        //Reshape output back
        output.Reshape({1, output.getshape()[0]});

//        output = (_weights * _local_input) + _biases;
    }

    void Linear::Backward(Array<double> &dLdY, Array<double> &dLdX) {


        // num rows in C
        int l = dLdX.getshape()[0];
        // num cols in C
        int n = dLdX.getshape()[1];
        // num cols in A
        int m = dLdY.getshape()[1];
        double alpha = 1.0;
        int lda = dLdY.getshape()[1];
        int ldb = _weights.getshape()[1];
        double beta = 0.0;
        int ldc = dLdX.getshape()[1];

        // do matmul
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, l, n,m,alpha, dLdY.getdata(), lda, _weights.getdata(), ldb, beta, dLdX.getdata(), ldc);


        // compute dLdX, this vector will be sent to be backpropagated through the previous layer
        //dLdX = dLdY * (_weights);

        dLdY.Reshape({dLdY.getshape()[1], 1});
        _local_input.Reshape({1, _local_input.getshape()[0]});
        // num rows in C
        l = _dLdW.getshape()[0];
        // num cols in C
        n = _dLdW.getshape()[1];
        // num cols in A
        m = dLdY.getshape()[1];
        alpha = 1.0;
        lda = dLdY.getshape()[1];
        ldb = _local_input.getshape()[1];
        beta = 1.0;
        ldc = _dLdW.getshape()[1];

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, l, n,m,alpha, dLdY.getdata(), lda, _local_input.getdata(), ldb, beta, _dLdW.getdata(), ldc);


        // compute gradients
        // this is dgemm
//        _dLdW += dLdY * _local_input;

        // add biases
        int len = _dLdB.getsize();
        double scalar = 1.0;
        cblas_daxpy(len, scalar, dLdY.getdata(), 1, _dLdB.getdata(), 1);
        // this is saxpy
//        _dLdB += dLdY;
    }

    template<typename Optimizer>
    void Linear::Update_Params(Optimizer *optimizer, size_t normalizer) {

        // update the biases and reset dLdB to zeros. MUST UPDATE BIASES FIRST or else member variable k of momentum optmizer
        // is prematurely updated
        (*optimizer).Forward(_biases, _dLdB, normalizer);
        _dLdB.fill(0);

        // update the weights and reset dLdW to zeros
        (*optimizer).Forward(_weights, _dLdW, normalizer);
        _dLdW.fill(0);

    }
}
#endif //ANN_LINEAR_CPP
