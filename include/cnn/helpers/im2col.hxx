//
// Created by Malav Patel on 1/14/23.
//

#ifndef CNN_EXAMPLE_IM2COL_HXX
#define CNN_EXAMPLE_IM2COL_HXX

#endif //CNN_EXAMPLE_IM2COL_HXX


#include "../lin_alg/array.hxx"

//! im2col converts an image into a matrix ready for convolution via matrix multiplication. This can take advantage
//! of optimized matrix multiplication routines included in BLAS.

// PARAMETERS
// im           | type Array<double>& |  image being converted to a matrix for convolutions
// col          | type Array<double>& |  matrix that data for matrix will be stored in
// N_zstr       | type const int |  number of strides along the z direction during convolution
// N_vstr       | type const int |  number of strides along the vertical (row, or height) direction
// N_hstr       | type const int |  number of strides along the horizontal (col, or width) direction
// N_maps       | type const int |  number of channels produced during each window slide
// filterheight | type const int | filter height (or number of rows in filter)
// filterwidth  | type const int | filter width (or number of cols in filter)
// v_           | type const int | vertical stride for the convolution
// h_           | type const int | horizontal stride for the convolution
// z_           | type const int | depth stride for the convolution

using namespace CNN;

void im2col( Array<double>& im, Array<double>& col, const int N_zstr, const int N_vstr, const int N_hstr, const int N_maps, const int filterheight, const int filterwidth,
             const int v_, const int h_, const int z_)
{
    // check that pre-allocation is correct
    assert(col.getsize() == N_zstr*N_vstr*N_hstr*N_maps*filterwidth*filterheight);
    assert(col.getshape().size() == 2);

    // check dimensions of image
    assert(im.getshape().size() == 3);

    // image width
    int n  = im.getshape()[2];
    int m = im.getshape()[1];

    // needs to be labeled later
    int stride = im.getstride()[0];
    int stride2 = filterwidth == 1 ? 0 : n;

    // strided array
    Array<double> strided = im.as_strided({N_zstr, N_vstr, N_hstr, N_maps, filterheight, filterwidth},
                                     {m*n*z_, n*v_,h_, stride, stride2, 1 });


    // pointers to data
    double* datacol = col.getdata();
    double* strided_data = im.getdata();

    // do im2col operation
    for (int zstr = 0 ; zstr < N_zstr; zstr++){
        for (int vstr = 0; vstr < N_vstr; vstr++) {
            for (int hstr = 0; hstr < N_hstr; hstr++) {
                for (int map = 0; map < N_maps; map++) {
                    for (int height = 0; height < filterheight; height++) {
                        for (int width = 0; width < filterwidth; width++) {
                            *(datacol++) = strided_data[zstr*m*n*z_ + vstr*n*v_ + hstr*h_ + map*stride + height*stride2 + width];
                                    //strided[{zstr, vstr, hstr, map, height, width}];
                        }
                    }
                }
            }
        }
    }
}