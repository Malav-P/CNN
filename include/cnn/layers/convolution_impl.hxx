//
// Created by malav on 6/17/2022.
//

#ifndef ANN_CONVOLUTION_IMPL_HXX
#define ANN_CONVOLUTION_IMPL_HXX

#include "convolution.hxx"
#include "../helpers/im2col.hxx"

namespace CNN {

    Convolution::Convolution(size_t in_maps, size_t out_maps, size_t in_width, size_t in_height, size_t filter_width,
                             size_t filter_height, size_t stride_h, size_t stride_v, size_t padleft, size_t padright,
                             size_t padtop,
                             size_t padbottom,
                             double *weights)
            : Layer(in_width, in_height, in_maps, 0, 0, 0),
              _h_str(stride_h),
              _v_str(stride_v),
              _padleft(padleft),
              _padright(padright),
              _padtop(padtop),
              _padbottom(padbottom),
              _filters({out_maps, in_maps, filter_height, filter_width}),
              _dLdFs({out_maps, in_maps, filter_height, filter_width})
              {

        // calculate total number of vertical and horizontal strides
        size_t num_v_strides = std::floor((_in.height + _padtop + _padbottom - filter_height) / _v_str) + 1;
        size_t num_h_strides = std::floor((_in.width + _padleft + _padright - filter_width) / _h_str) + 1;

        // specify output shape
        _out = {num_h_strides, num_v_strides, out_maps};


        // Glorot initialize the weights if weights pointer is null
        if (weights == nullptr) {

            // get the current time to seed the random number generator
            typedef std::chrono::high_resolution_clock myclock;
            myclock::time_point beginning = myclock::now();
            myclock::duration d = myclock::now() - beginning;
            unsigned seed2 = d.count();

            size_t m = _in.height + _padtop + _padbottom;
            size_t n = _in.width + _padleft + _padright;

            // seed the random number generator
            std::default_random_engine generator(seed2);
            std::uniform_real_distribution<double> distribution(-sqrt(6.0 / (m * n + _out.height * _out.width)),
                                                                sqrt(6.0 / (m * n + _out.height * _out.width)));

            double* data = _filters.getdata();
            for (int i = 0 ; i < _filters.getsize(); i++)
            {
                *(data++) = distribution(generator);
            }
        }

            // if weights pointer is provided then we fill in the values
        else {
            std::memcpy(_filters.getdata(), weights, _filters.getsize() * sizeof(double));

        }


    }

    Convolution::Convolution(size_t in_maps, size_t out_maps, size_t in_width, size_t in_height, size_t filter_width,
                             size_t filter_height, size_t stride_h, size_t stride_v, bool padding, double *weights)
            : _h_str(stride_h),
              _v_str(stride_v),
              _filters({out_maps, in_maps, filter_height, filter_width}),
              _dLdFs({out_maps, in_maps, filter_height, filter_width}),
              Layer(in_width, in_height, in_maps, 0, 0, 0) {
        if (!padding)
        {
            _padleft = 0; _padright = 0; _padtop = 0; _padbottom = 0;
        }

        else
        {
            // total number of padded rows
            size_t vert_pad = (in_height - 1) * stride_v - in_height + filter_height;

            // number of padded rows at top of matrix
            _padtop = std::floor(vert_pad / 2.0);

            // number of padded rows at bottom of matrix
            _padbottom = std::ceil(vert_pad / 2.0);

            // total number of padded columns
            size_t hor_pad = (in_width - 1) * stride_h - in_width + filter_width;

            // number of padded columns to left of matrix
            _padleft = std::floor(hor_pad / 2.0);

            // number of padded columns to right of matrix
            _padright = std::ceil(hor_pad / 2.0);
        }

        // input height with padding
        size_t m = _in.height + _padtop + _padbottom;
        // input width with padding
        size_t n = _in.width + _padleft + _padright;

        // calculate total number of vertical and horizontal strides
        size_t num_v_strides = std::floor((m - filter_height) / _v_str) + 1;
        size_t num_h_strides = std::floor((n - filter_width) / _h_str) + 1;

        // output shape
        _out = {num_h_strides, num_v_strides, out_maps};

        // Glorot initialize the weights if weights pointer is null
        if (weights == nullptr)
        {

            // get the current time to seed the random number generator
            typedef std::chrono::high_resolution_clock myclock;
            myclock::time_point beginning = myclock::now();
            myclock::duration d = myclock::now() - beginning;
            unsigned seed2 = d.count();

            // seed the random number generator
            std::default_random_engine generator(seed2);
            std::uniform_real_distribution<double> distribution(-sqrt(6.0 / (m * n + _out.height * _out.width)),
                                                                sqrt(6.0 / (m * n + _out.height * _out.width)));


            double* data = _filters.getdata();
            for (int i = 0 ; i < _filters.getsize(); i++)
            {
                *(data++) = distribution(generator);
            }
        }


        // if weights pointer is provided then we fill in the values
        else
        {
            std::memcpy(_filters.getdata(), weights, _filters.getsize() * sizeof(double));
        }

    }

    void Convolution::Forward(Array<double> &input, Array<double> &output) {
        // note that input length matching with _in parameters is indirectly checked in the matrix*vector operator overload

        // this routine can be optimized (we take a vector, turn it into matrix, pad it, then flatten back to vector)
        // find a way to do the padding with the vector itself
        output.Reshape({_out.depth, _out.height, _out.width});

        _local_input = Array<double>(input);
        _local_input.Reshape({_in.depth, _in.height, _in.width});
        _local_input = _local_input.pad({0, 0, _padtop, _padbottom, _padleft, _padright});


        // generate col matrix using im2col
        int filterwidth = _filters.getshape()[3];
        int filterheight = _filters.getshape()[2];
        int filterdepth = _filters.getshape()[1];

        int N_zstr = 1;
        int N_vstr = _out.height;
        int N_hstr = _out.width;

        int v_ = _v_str;
        int h_ = _h_str;
        int z_ = 1;

        // TODO - shape dimensions should actaully be swapped
        Array<double> col({ filterwidth*filterheight*filterdepth, _out.width*_out.height});

        im2col(_local_input, col,N_zstr, N_vstr, N_hstr, filterdepth, filterheight, filterwidth, v_, h_, z_);


        // num rows in C
        int l = _out.depth;
        // num cols in C
        int n = _out.width*_out.height;
        // num cols in A
        int m = filterwidth*filterheight*filterdepth;
        double alpha = 1.0;
        int lda = m;
        int ldb = m;
        double beta = 0.0;
        int ldc = _out.width*_out.height;

        // do matmul
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, l, n,m,alpha, _filters.getdata(), lda, col.getdata(), ldb, beta, output.getdata(), ldc);

        output.Reshape({1, output.getsize()});
    }


    void Convolution::Backward(Array<double> &dLdYs, Array<double> &dLdXs) {

        int p = _in.height + _padtop + _padbottom;
        int q = _in.width + _padleft + _padright;

        int filter_height = _filters.getshape()[2];
        int filter_width = _filters.getshape()[3];
        int filter_depth = _in.depth;

        //! begin overhaul

        Array<double> reform_output({_out.depth, _out.height + ((p - filter_height) % _v_str) + (_out.height - 1) * (_v_str - 1),
                                     _out.width + ((q - filter_width) % _h_str) + (_out.width - 1) * (_h_str - 1)});

        double* reform_output_data = reform_output.getdata();
        vector<int> reform_output_strides = reform_output.getstride();

        dLdYs.Reshape({_out.depth, _out.height, _out.width});
        double* dLdYs_data = dLdYs.getdata();
        vector<int> dLdYs_strides = dLdYs.getstride();

        for (int k = 0; k < _out.depth; k++)
        {
            for (int i = 0; i < _out.height; i++)
            {
                for (int j = 0; j < _out.width; j++)
                {
                    reform_output_data[k*reform_output_strides[0] + i*_v_str*reform_output_strides[1] + j*_h_str*reform_output_strides[2]] = dLdYs_data[k*dLdYs_strides[0] + i*dLdYs_strides[1] + j*dLdYs_strides[2]];
                }
            }
        }

        // generate col matrix using im2col
        int reformoutputwidth = reform_output.getshape()[2];
        int reformoutputheight = reform_output.getshape()[1];
        int reformoutputdepth = reform_output.getshape()[0];

        // convolve the input images with reformatted output with unit strides
        int N_zstr = _in.depth;
        int N_vstr = std::floor((p - reform_output.getshape()[1])) + 1;
        int N_hstr = std::floor((q - reform_output.getshape()[2])) + 1;

        int N_maps = 1;

        int v_ = 1;
        int h_ = 1;
        int z_ = 1;

        // TODO - shape dimensions should actually be swapped
        Array<double> col({ reformoutputheight*reformoutputwidth*N_maps, N_zstr*N_vstr*N_hstr});

        im2col(_local_input, col, N_zstr, N_vstr, N_hstr, N_maps, reformoutputheight, reformoutputwidth, v_, h_, z_);

        // num rows in C, also num rows in A
        int l = reformoutputdepth;
        // num cols in C, also num cols in B
        int n = filter_width*filter_height*filter_depth;
        // num cols in A
        int m = reformoutputwidth*reformoutputheight*N_maps;
        double alpha = 1.0;
        int lda = m;
        int ldb = m;
        double beta = 1.0;
        int ldc = n;

        // do matmul
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, l, n,m,alpha, reform_output.getdata(), lda, col.getdata(), ldb, beta, _dLdFs.getdata(), ldc);

        // this concludes the calculation of _dLdFs

        // we move to calculation of dLdX

        // add padding to reformatted matrix
        reform_output = reform_output.pad({0,0,filter_height - 1 - _padtop, filter_height - 1 - _padbottom, filter_width - 1 - _padleft, filter_width - 1 - _padright});

        N_zstr = _out.depth;
        N_vstr = _in.height;
        N_hstr = _in.width;

        v_ = 1;
        h_ = 1;
        z_ = 1;

        N_maps = 1;

        // TODO - shape dimensions should actaully be swapped
        col.Reshape({ filter_width*filter_height*N_maps, N_zstr*N_vstr*N_hstr});

        im2col(reform_output, col, N_zstr, N_vstr, N_hstr, N_maps, filter_height, filter_width, v_, h_, z_);

        // num rows in C, depth of each filter
        l = filter_depth;
        // num cols in C, total number of strides for one slice of reformoutput
        n = N_vstr*N_hstr;
        // num cols in A
        m = filter_width*filter_height;
        alpha = 1.0;
        lda = m;
        ldb = m;
        beta = 1.0;
        ldc = n;

        Array<double> rotated_filters = _filters.rotate();
        for (int idx = 0 ; idx < _out.depth ; idx++)
        {
             double* rot_filter_data = rotated_filters.getdata() + idx*rotated_filters.getstride()[0];
             double* col_data = col.getdata() + idx*N_vstr*N_hstr*filter_height*filter_width;

            // do matmul
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, l, n,m,alpha, rot_filter_data, lda, col_data, ldb, beta, dLdXs.getdata(), ldc);
        }

        // we are averaging the loss gradient over the total number of filters
        // alternatively, change alpha on line 297 to 1/N_filters
        dLdXs *= 1.0 / _out.depth;

        // reshape operation technically not needed
        dLdXs.Reshape({1, dLdXs.getsize()});

        //! END OVERHAUL

    }

    template<typename Optimizer>
    void Convolution::Update_Params(Optimizer *optimizer, size_t normalizer) {

            // update the weights according to the optimizer
            (*optimizer).Forward(_filters, _dLdFs, normalizer);

            // fill the gradient with zeros, ideally could use memset
            _dLdFs.fill(0);
    }


} // namespace CNN
#endif //ANN_CONVOLUTION_IMPL_HXX
