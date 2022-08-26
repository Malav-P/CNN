//
// Created by malav on 4/28/2022.
//

#ifndef ANN_MAT_IMPL_HXX
#define ANN_MAT_IMPL_HXX

#include "mat.hxx"
#include "vector.hxx"

//! constructor -------------------------------------------------------------------------
template<typename T>
Mat<T>::Mat(size_t side_len, T* arr)
: _rows(side_len)
, _cols(side_len)
, _data(new T[side_len * side_len]{0})
{
    if(arr != nullptr)
    {
        // TODO : currently no way to ensure arr has side_len*side_len elements in it
        std::memcpy(_data, arr, sizeof(T) * _rows * _cols);
    }
}
//! --------------------------------------------------------------------------------------

//! constructor --------------------------------------------------------------------------
template<typename T>
Mat<T>::Mat(size_t num_r, size_t num_c, T* arr)
: _rows(num_r)
, _cols(num_c)
, _data(new T[num_r * num_c]{0})
{
    if(arr != nullptr)
    {
        // TODO : currently no way to ensure arr has num_r*num_c elements in it
        std::memcpy(_data, arr, sizeof(T) * _rows * _cols);
    }
}
//!--------------------------------------------------------------------------------------

//! copy constructor --------------------------------------------------------------------
template<typename T>
Mat<T>::Mat(const Mat<T>& other)
: _rows(other._rows)
, _cols(other._cols)
, _rot(other._rot)
, _data(new T[_rows*_cols]{0})
{
    std::memcpy(_data, other._data, sizeof(T) * _rows * _cols);
}
//! ---------------------------------------------------------------------------------------
//! move constructor -------------------------------------------------------------
template<typename T>
Mat<T>::Mat(Mat<T> &&other) noexcept
        : _data(other._data)
        , _rows(other._rows)
        , _cols(other._cols)
        , _rot(other._rot)
{
    // remove pointer to other data
    other._data = nullptr;
}
//! -------------------------------------------------------------


//! copy assignment operator--------------------------------------------------------------------------------------
template<typename T>
Mat<T>& Mat<T>::operator=(const Mat<T>& rhs){

    // check for self-assignment
    if(this != &rhs)
    {
        // make copy of input
        Mat<T> tmp(rhs);

        // swap values with tmp variable
        std::swap(_cols, tmp._cols);
        std::swap(_rows, tmp._rows);
        std::swap(_rot, tmp._rot);

        // no need to reallocate if same size
        if (_rows * _cols != tmp._cols * tmp._rows)
        {
            delete[] _data;
            _data = new T[_rows * _cols];
        }

        // copy lin_alg to new pointer
        //  equivalent to   for (size_t k = 0; k < _rows * _cols; k++) { _data[k] = tmp._data[k]; }
        std::memcpy(_data, tmp._data, sizeof(T) * _rows * _cols);
    }

    // return result
    return (*this);
}
//! -------------------------------------------------------------------------------------------------


//! ---------------------------------------------------------------------------
template<typename T>
Mat<T> &Mat<T>::operator=(Mat<T> &&other)  noexcept {
    if (this != &other)
    {
        // Free the existing resource.
        delete[] _data;

        // Copy the _data pointer and its _length from the
        // source object.
        _data   = other._data;
        _rows   = other._rows;
        _cols   = other._cols;
        _rot    = other._rot;

        // Release the _data pointer from the source object so that
        // the destructor does not free the memory multiple times.
        other._data = nullptr;
    }
    return *this;
}
//! ---------------------------------------------------------------------------

//! += operator----------------------------------------------------------------
template<typename T>
Mat<T> &Mat<T>::operator+=(const Mat<T> &other)
{
    // assert that RHS and LHS have same dimensions
    assert(_rows == other._rows && _cols == other._cols);

    // do += operation
    for (size_t i = 0; i < _rows ; i++){ for (size_t j = 0; j < _cols ; j++){
            (*this)(i,j) += other(i,j);
        }}

    // return the current object
    return (*this);
}
//! ------------------------------------------------------------------------------
//! += operator----------------------------------------------------------------
template<typename T>
Mat<T> Mat<T>::operator+(const Mat<T> &rhs)
{

    // assert that matrix dimensions are the same
    assert(_rows == rhs._rows && _cols == rhs._cols);

    // initialize return variable
    Mat<T> obj(_rows ,_cols);

    // do + operation
    for (size_t i = 0; i < _rows ; i++){ for (size_t j = 0; j < _cols ; j++){
            obj(i,j)  = rhs(i,j) + (*this)(i,j);
        }}

    // return the current object
    return obj;
}
//! ------------------------------------------------------------------------------
//! matrix index operator myObj(i,j) returns the i-jth element of matrix ------------------
template<typename T>
T& Mat<T>::operator()(size_t i, size_t j) { return _data[i * _cols + j]; }
//! ---------------------------------------------------------------------------------------

//! same as above but for const objects ---------------------------------------------------
template<typename T>
const T& Mat<T>::operator()(size_t i, size_t j) const { return _data[i * _cols + j];}
//! ---------------------------------------------------------------------------------------

//! multiply two matrices together, overloading the multiplication operator ---------------
template<typename T>
Mat<T> Mat<T>::operator*(const Mat<T>& other)
{
    // check to make sure matrix sizes are compatible
    assert(_cols == other._rows);

    // initialize return variable
    Mat<T> obj(_rows, other._cols);

    // do matrix multiplication
    for (size_t i = 0; i < obj._rows ; i++)
    {
        for (size_t j = 0; j < obj._cols; j++)
        {
            for (size_t k = 0; k < _cols; k++)
            {
                obj(i,j) += (*this)(i, k) * other(k,j);
            }
        }
    }

    // return result
    return obj;
}
//! -----------------------------------------------------------------------------------
//! multiply matrix with scalar --------------------------------------------------------
template<typename T>
Mat<T> Mat<T>::operator*(const double c)
{

    // initialize return variable
    Mat<T> obj(_rows,_cols);

    // do * operation
    for (size_t i = 0; i < _rows ; i++){ for (size_t j = 0; j < _cols ; j++){
            obj(i,j)  = c * (*this)(i,j);
        }}

    // return the current object
    return obj;
}
//! -----------------------------------------------------------------------------------
//! multiply matrix with col vector ---------------------------------------------------
template<typename T>
Vector<T> Mat<T>::operator * (const Vector<T>& other)
{
    // check to make sure matrix is compatible with vector
    assert(_cols == other._length);

    // initialize return variable
    Vector<T> obj(_rows);

    // do multiplication
    for (size_t i = 0; i < _rows; i++)
    {
        obj[i] = 0;
        for (size_t j = 0; j < _cols; j++)
        {
            obj[i] += other[j] * (*this)(i, j);
        }
    }

    // return result
    return obj;
}
//! ----------------------------------------------------------------------------------------

//! compute dot product between two matrices -----------------------------------------------
template<typename T>
T Mat<T>::dot(const Mat<T>& other)
{
    // ensure matrices are same size
    assert((other._cols == _cols) && (other._rows == _rows));

    // initialize return variable
    T answer = 0;

    // do dot product
    for (size_t k = 0; k < _rows * _cols; k++) { answer += _data[k] * (other._data)[k]; }

    // return result
    return answer;
}
//! -------------------------------------------------------------------------------------------

//! compute dot product between overlapping parts of matrices ---------------------------------
template<typename T>
T Mat<T>::partial_dot(const Mat<T>& other, Dims p)
{
    // starting indices must be within bounds of matrix
    assert(p.height < _rows && p.width < _cols);

    // overlapping matrix must be within bounds of this matrix
    assert(p.width + other._rows <= _rows && p.height + other._cols <= _cols);

    // initialize return variable;
    T answer = 0;

    // do partial dot operation
    for (size_t i = 0; i < other._rows ; i++) { for (size_t j = 0; j< other._cols ; j++)
        {
            answer += other(i,j) * (*this)(p.width + i, p.height + j);
        }}

    // return result
    return answer;
}
//! -----------------------------------------------------------------------------------------

//! fill matrix with value -------------------------------------------------------------------
template<typename T>
void Mat<T>::fill(T t) { for (size_t i=0; i < _cols * _rows; i++) { _data[i] = t;} }
//! ------------------------------------------------------------------------------------------

//! set rotation state of matrix ------------------------------------------------------------
template<typename T>
void Mat<T>::set_rot(size_t n)
{
    // rotation state can be either 0, 1, 2, or 3
    n %= 4;

    // need to rotate clockwise
    if (_rot < n)
    {
        size_t diff = n - _rot;
        for (size_t i = 0; i < diff ; i++) {rotate_once();} // clockwise rotation is good here
    }
    // need to rotate counterclockwise
    else if (_rot > n)
    {
        size_t diff = 4 - (_rot - n);
        for (size_t i = 0; i < diff ; i++) {rotate_once();} // eventually change to rotate CCW, it is more efficient
    }
    // Nothing to do, _rot == n
    else
    {}
}
//! ---------------------------------------------------------------------------------------------
//! pad matrix with zeros---------------------------------------------------------------------------
template<typename T>
void Mat<T>::padding(size_t padleft, size_t padright, size_t padtop, size_t padbottom)
{
    // copy data to local variable
    T tmp[_rows * _cols];
    std::memcpy(tmp, _data, sizeof(T) * _rows *_cols);


    // increase number of rows and cols
    size_t _rows_new = _rows + padtop + padbottom;
    size_t _cols_new = _cols + padleft + padright;

    // reallocate memory
    delete[] _data;
    _data = new T[_rows_new * _cols_new]{0};


    // do padding operation
    for (size_t i = padtop; i < _rows_new - padbottom; i++)
    {
        for (size_t j = padleft; j < _cols_new - padright; j++)
        {
            _data[i * _cols_new + j] = tmp[(i - padtop) * (_cols) + (j - padleft)];
        }
    }

    // change member variable states
    _rows += padtop + padbottom;
    _cols += padleft + padright;
}
//! flatten matrix into vector ------------------------------------------------------------------
template<typename T>
Vector<T> Mat<T>::flatten()
{
    // initialize return variable
    Vector<T> obj(_rows * _cols);

    // do flatten operation
    for (size_t i = 0; i < _rows * _cols; i++)
    { std::memcpy(obj._data, _data, sizeof(T) * _rows*_cols);}

    // return result
    return obj;
}
//! --------------------------------------------------------------------------------------------

//! transpose the matrix ------------------------------------------------------------------
template<typename T>
Mat<T> Mat<T>::transpose()
{
    // initialize return variable
    Mat<T> obj(_cols, _rows);

    // do transpose operation
    for(size_t i=0; i < obj._rows ; i++)
    {
        for (size_t j=0; j < obj._cols; j++) { obj._data[i * obj._cols + j] = _data[j * _cols + i]; }
    }

    // return result
    return obj;
}
//! --------------------------------------------------------------------------------------
template<typename T>
void Mat<T>::keep(Dims *indices)
{
    Dims* current = indices;

    Dims candidate;

    for (size_t i = 0; i<_rows; i++) {for (size_t j = 0; j < _cols ; j++) {
        candidate = {i,j};

        if (candidate == *current) {
            current += 1;
        }

        else {(*this)(i,j) = 0;}
    }}
}
//! ------------------------------------------------------------------------------------

//! rotate matrix once (helper for set_rotate) -------------------------------------------------
template<typename T>
void Mat<T>::rotate_once()
{
    // copy lin_alg to local variable
    T tmp[_rows * _cols];
    for (size_t k = 0; k < _rows * _cols; k++) { tmp[k] = _data[k];}

    // swap rows and cols
    std::swap(_rows, _cols);

    // do the rotation operation
    for (size_t new_i = 0; new_i < _rows; new_i++)
    {
        for (size_t new_j=0; new_j < _cols; new_j++)
        {
            _data[new_i * _cols + new_j] = tmp[(_cols - 1 - new_j) * _rows + new_i];
        }
    }

    // change member variable state
    _rot += 1;

    // rotation state can be one of 0, 1, 2, or 3
    _rot %= 4;
}


//! ---------------------------------------------------------------------------------------

#endif //ANN_MAT_IMPL_HXX
