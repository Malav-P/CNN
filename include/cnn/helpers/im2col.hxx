//
// Created by Malav Patel on 1/14/23.
//

#ifndef CNN_EXAMPLE_IM2COL_HXX
#define CNN_EXAMPLE_IM2COL_HXX

#endif //CNN_EXAMPLE_IM2COL_HXX


#include "../lin_alg/array.hxx"

using namespace CNN;


void im2col( Array<double>& im, Array<double>& col, const int N_zstr, const int N_vstr, const int N_hstr, const int N_maps, const int filterheight, const int filterwidth,
             const int v_, const int h_, const int z_)
{
    // check that preallocation is correct
    assert(col.getsize() == N_zstr*N_vstr*N_hstr*N_maps*filterwidth*filterheight);
    assert(col.getshape().size() == 2);

    // check dimensions of image
    assert(im.getshape().size() == 3);

    double* datacol = col.getdata();
    double* strided_data = im.getdata();

    // image width
    int n  = im.getshape()[2];
    int m = im.getshape()[1];

    // needs to be labeled later
    int stride = im.getstride()[0];
    int stride2 = filterwidth == 1 ? 0 : n;

    Array<double> strided = im.as_strided({N_zstr, N_vstr, N_hstr, N_maps, filterheight, filterwidth},
                                     {m*n*z_, n*v_,h_, stride, stride2, 1 });

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