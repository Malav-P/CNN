//
// Created by malav on 10/11/2022.
//

#ifndef CNN_LIN_ALG_KERNELS_HXX
#define CNN_LIN_ALG_KERNELS_HXX

#include "mat.hxx"
#include "vector.hxx"

__global__
void vecMat_Kernel( double* d_in, Mat<double>* d_matrix, double* d_out)
{
    size_t N_ROWS = d_matrix->get_rows();
    size_t N_COLS = d_matrix->get_cols();

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < N_COLS)
    {
        for (size_t i = 0 ; i < N_ROWS ; i++)
        {
            d_out[j] += (*d_matrix)(i,j) * d_in[i];
        }
    }
}

__global__
void matVec_Kernel(Mat<double>* d_matrix, double* d_in, double* d_out)
{
    size_t N_ROWS = d_matrix->get_rows();
    size_t N_COLS = d_matrix->get_cols();

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N_ROWS)
    {
        for (size_t j = 0 ; j < N_COLS ; j++)
        {
            d_out[i] += (*d_matrix)(i,j) * d_in[j];
        }
    }
}

__global__
void vecVecplusequals_Kernel(Mat<double>* d_matrix, double* d_1, double* d_2)
{
    size_t N_ROWS = d_matrix->get_rows();
    size_t N_COLS = d_matrix->get_cols();

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N_ROWS && j < N_COLS)
    {
        (*d_matrix)(i,j) += d_1[i] * d_2[j];
    }
}

__global__
void plus_equals_Kernel(size_t N, double* d_L, double* d_R, double c)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        d_L[i] += c*d_R[i];
    }
}

__global__
void fill_Kernel(size_t N, double* d_data, double fill)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        d_data[i] = fill;
    }
}

#endif //CNN_LIN_ALG_KERNELS_HXX
