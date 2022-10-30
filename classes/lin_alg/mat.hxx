//
// Created by malav on 4/26/2022.
//

#ifndef ANN_MAT_HXX
#define ANN_MAT_HXX

#include "data_types.hxx"
#include "vector.hxx"


template<typename T = double>
class Mat {
    public:

        //! CONSTRUCTORS, DESTRUCTORS, MOVE CONSTRUCTORS, ETC --------------------------------------------------------

        // default constructor
        Mat() = default;

        // constructor
        explicit Mat(size_t side_len, T* arr = nullptr);

        // constructor
        __host__ __device__ Mat(size_t num_r, size_t num_c, T* arr = nullptr);

        // copy constructor
        Mat(const Mat<T>& other);

        // move constructor
        Mat(Mat<T>&& other) noexcept;

        // destructor
        __host__ __device__ ~Mat() { delete[] _data; }

        //! --------------------------------------------------------------------------------------------------------

        //! ASSIGNMENT, BINARY, UNARY OPERATORS -------------------------------------------------------------------
        // copy assignment operator
        __host__ __device__ Mat<T>& operator=(const Mat<T>& rhs);

        // move assignment operator
        __host__ __device__ Mat<T>& operator=(Mat<T>&& other) noexcept ;

        // += operator
        Mat<T>& operator+=(const Mat<T>& other);

        // + operator
        Mat<T> operator+(const Mat<T>& rhs);

        // index operator
        __host__ __device__ T& operator()(size_t i, size_t j);

        // const index operator
        __host__ __device__ const T& operator()(size_t i, size_t j) const;

        // multiply operator (matrix)
        Mat<T> operator * (const Mat<T>& other);

        // multiply operator (scalar)
        Mat<T> operator * (double c);

        // multiply operator (vector)
        Vector<T> operator * (const Vector<T>& other);

        //! -------------------------------------------------------------------------------------------------------

        //! OTHER -------------------------------------------------------------------------------------------------

        // compute inner product between two Matrices
        T dot(const Mat<T>& other);

        // compute partial inner product with given Dims of starting indices
        __host__ __device__ T partial_dot(const Mat<T>& other, Dims p);

        // keep values in certain indices, and set all others to zero. indices must be sorted in increasing row, increasing column order
        void keep(Dims* indices);

        // fill matrix with a value
        void fill(T t);

        // set the rotation state of the matrix and restructure _data
        __host__ __device__ void set_rot(size_t n = 0);

        // add padding to matrix
        void padding(size_t padleft, size_t padright, size_t padtop, size_t padbottom);

        // crop the matrix/ remove padded rows from matrix
        void crop(size_t crop_left, size_t crop_right, size_t crop_top, size_t crop_bottom);

        // create a Vector object out of this object
        Vector<T> flatten();

        // transpose the matrix
        Mat<T> transpose();

        // get number of rows of matrix (read only)
        __host__ __device__ size_t const& get_rows() const {return _rows;}

        // get number of cols of matrix (read only)
        __host__ __device__ size_t const& get_cols() const {return _cols;}

        // get rotation state
        size_t const& get_rot()  const {return _rot;}

        // port data to GPU
        void port_to_GPU(Mat<T>*& d_mat, T*& d_mat_data);

        // print the matrix
        void print() const;

        //! --------------------------------------------------------------------------------------------------------


    private:

        // number of columns
        size_t _rows {0};

        // number of rows
        size_t _cols {0};

        // rotation state, 0 = no rotate; 1 = 90 degrees clockwise; 2 = 180 degrees clockwise; 3 = 270 degrees clockwise
        size_t _rot {0};

        // helper for set_rot, rotate matrix CW by 90 degrees
        __host__ __device__ void rotate_once();

        friend class Vector<T>;
    public:
            // pointer to _data
            T* _data {nullptr};

};



#include "mat_impl.hxx"
#endif //ANN_MAT_HXX
