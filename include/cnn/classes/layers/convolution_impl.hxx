//
// Created by malav on 6/17/2022.
//

#ifndef ANN_CONVOLUTION_IMPL_HXX
#define ANN_CONVOLUTION_IMPL_HXX

#include "convolution.hxx"
#include "../lin_alg/miscellaneous_helpers.hxx"


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
              _filters(out_maps),
              _dLdFs(out_maps),
              _local_input(in_maps){

        // calculate total number of vertical and horizontal strides
        size_t num_v_strides = std::floor((_in.height + _padtop + _padbottom - filter_height) / _v_str) + 1;
        size_t num_h_strides = std::floor((_in.width + _padleft + _padright - filter_width) / _h_str) + 1;

        // specify output shape
        _out = {num_h_strides, num_v_strides, out_maps};

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

        // allocate memory for an initialize filters
        for (size_t i = 0; i < _filters.size(); i++) {
            _filters[i] = Cuboid<double>(filter_height, filter_width, in_maps);
            _dLdFs[i] = Cuboid<double>(filter_height, filter_width, in_maps);
        }

        // Glorot initialize the weights if weights pointer is null
        if (weights == nullptr) {
            for (Cuboid<double> &_filter: _filters) {
                for (size_t i = 0; i < filter_height; i++) {
                    for (size_t j = 0; j < filter_width; j++) {
                        for (size_t k = 0; k < in_maps; k++) {
                            _filter(i, j, k) = distribution(generator);
                        }
                    }
                }
            }
        }

            // if weights pointer is provided then we fill in the values
        else {
            size_t len = _filters[0].get_depth() * _filters[0].get_rows() * _filters[0].get_cols();
            for (size_t i = 0; i < _filters.size(); i++) {
                std::memcpy(_filters[i].get_data(), weights + (i * len), len * sizeof(double));
            }
        }


    }

    Convolution::Convolution(size_t in_maps, size_t out_maps, size_t in_width, size_t in_height, size_t filter_width,
                             size_t filter_height, size_t stride_h, size_t stride_v, bool padding, double *weights)
            : _h_str(stride_h),
              _v_str(stride_v),
              _filters(out_maps),
              _dLdFs(out_maps),
              _local_input(in_maps),
              Layer(in_width, in_height, in_maps, 0, 0, 0) {
        if (!padding) {
            _padleft = 0;
            _padright = 0;
            _padtop = 0;
            _padbottom = 0;

        } else {
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


        size_t m = _in.height + _padtop + _padbottom;
        size_t n = _in.width + _padleft + _padright;

        // calculate total number of vertical and horizontal strides
        size_t num_v_strides = std::floor((m - filter_height) / _v_str) + 1;
        size_t num_h_strides = std::floor((n - filter_width) / _h_str) + 1;

        // specify output shape
        _out = {num_h_strides, num_v_strides, out_maps};

        // get the current time to seed the random number generator
        typedef std::chrono::high_resolution_clock myclock;
        myclock::time_point beginning = myclock::now();
        myclock::duration d = myclock::now() - beginning;
        unsigned seed2 = d.count();

        // seed the random number generator
        std::default_random_engine generator(seed2);
        std::uniform_real_distribution<double> distribution(-sqrt(6.0 / (m * n + _out.height * _out.width)),
                                                            sqrt(6.0 / (m * n + _out.height * _out.width)));

        // initialize filters
        for (size_t i = 0; i < _filters.size(); i++) {
            _filters[i] = Cuboid<double>(filter_height, filter_width, in_maps);
            _dLdFs[i] = Cuboid<double>(filter_height, filter_width, in_maps);
        }

        // Glorot initialize the weights if no weight data is provided
        if (weights == nullptr){
            for (Cuboid<double> &_filter: _filters) {
                for (size_t i = 0; i < filter_height; i++) {
                    for (size_t j = 0; j < filter_width; j++) {
                        for (size_t k = 0; k < in_maps; k++) {
                            _filter(i, j, k) = distribution(generator);
                        }
                    }
                }
            }
        }


        // if weights pointer is provided then we fill in the values
        else {
            size_t len = _filters[0].get_depth() * _filters[0].get_rows() * _filters[0].get_cols();
            for (size_t i = 0; i < _filters.size(); i++) {
                std::memcpy(_filters[i].get_data(), weights + (i * len), len * sizeof(double));
            }
        }

    }

    void Convolution::Forward(Vector<double> &input, Vector<double> &output) {
        // note that input length matching with _in parameters is indirectly checked in the matrix*vector operator overload

        // this routine can be optimized (we take a vector, turn it into matrix, pad it, then flatten back to vector)
        // find a way to do the padding with the vector itself

        size_t rows = _in.height;
        size_t cols = _in.width;

        for (size_t i = 0; i < _filters[0].get_depth(); i++) {
            _local_input[i] = Mat<double>(rows, cols, input.get_data() + i * rows * cols);
            _local_input[i].padding(_padleft, _padright, _padtop, _padbottom);
        }

        Cuboid<double> input_cube = cubify(_local_input);

        for (size_t k = 0; k < _filters.size(); k++) {

            // do convolution
            for (size_t i = 0; i < _out.height; i++) {
                for (size_t j = 0; j < _out.width; j++) {
                    output[k * _out.height * _out.width + i * _out.width + j] = input_cube.partial_dot(_filters[k],
                                                                                                       {i * _v_str,
                                                                                                        j * _h_str, 0});
                }
            }
        }
    }


    void Convolution::Backward(Vector<double> &dLdYs, Vector<double> &dLdXs) {

        size_t m = _out.height;
        size_t n = _out.width;

        size_t p = _in.height + _padtop + _padbottom;
        size_t q = _in.width + _padleft + _padright;

        size_t filter_height = _filters[0].get_rows();
        size_t filter_width = _filters[0].get_cols();

        size_t N_filters = _filters.size();
        size_t N_in_maps = _filters[0].get_depth();


        for (size_t idx = 0; idx < N_filters; idx++) {
            // reshape dLdY into a matrix
            double *dLdY = dLdYs.get_data() + idx * m * n;

            // reformatted output
            Mat<double> reformatted_output(m + ((p - filter_height) % _v_str) + (m - 1) * (_v_str - 1),
                                           n + ((q - filter_width) % _h_str) + (n - 1) * (_h_str - 1));

            // fill in reformatted output matrix with the correct values
            for (size_t i = 0; i < m; i++) {
                for (size_t j = 0; j < n; j++) {
                    reformatted_output(i * (_v_str), j * (_h_str)) = dLdY[i * n + j];
                }
            }

            // convolve the input images with reformatted output with unit strides
            size_t num_v_strides = std::floor((p - reformatted_output.get_rows())) + 1;
            size_t num_h_strides = std::floor((q - reformatted_output.get_cols())) + 1;

            for (size_t k = 0; k < N_in_maps; k++) {
                for (size_t i = 0; i < num_v_strides; i++) {
                    for (size_t j = 0; j < num_h_strides; j++) {
                        _dLdFs[idx](i, j, k) += _local_input[k].partial_dot(reformatted_output, {i, j});
                    }
                }
            }

            // this concludes the calculation of _dLdFs

            // we move to calculation of dLdX

            // add padding to reformatted matrix
            reformatted_output.padding(filter_width - 1, filter_width - 1, filter_height - 1, filter_height - 1);

            num_v_strides = std::floor((reformatted_output.get_rows() - filter_height)) + 1;
            num_h_strides = std::floor((reformatted_output.get_cols() - filter_width)) + 1;

            // number of strides in each direction should be equal to the dimensions of dLdX_matrix
            assert(num_v_strides == p);
            assert(num_h_strides == q);

            size_t n_rows = num_v_strides - _padtop - _padbottom;
            size_t n_cols = num_h_strides - _padleft - _padright;

            for (size_t k = 0; k < N_in_maps; k++) {
                Mat<double> filter_plane(filter_height, filter_width,
                                         _filters[idx]._data + k * filter_width * filter_height);
                //rotate filter by 180 degrees
                filter_plane.set_rot(2);
                // crop the matrices and only look at cropped portion of data
                for (size_t i = 0; i < n_rows; i++) {
                    for (size_t j = 0; j < n_cols; j++) {
                        dLdXs[k * n_rows * n_cols + (i) * n_cols + (j)] += reformatted_output.partial_dot(filter_plane,
                                                                                                          {i + _padtop,
                                                                                                           j +
                                                                                                           _padleft});
                    }
                }
            }

        }
        // we are averaging the loss gradient over the total number of filters
        dLdXs *= 1.0 / N_filters;
    }

    template<typename Optimizer>
    void Convolution::Update_Params(Optimizer *optimizer, size_t normalizer) {
        for (size_t i = 0; i < _filters.size(); i++) {
            // update the weights according to the optimizer
            (*optimizer).Forward(_filters[i], _dLdFs[i], normalizer);

            // fill the gradient with zeros
            _dLdFs[i].fill(0);
        }

    }

    void Convolution::print_filters() {
        for (size_t i = 0; i < _filters.size(); i++) {
            std::cout << "FILTER " << i << "----------------\n\n";
            _filters[i].print();
        }
    }

} // namespace CNN
#endif //ANN_CONVOLUTION_IMPL_HXX
