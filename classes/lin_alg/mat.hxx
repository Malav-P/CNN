//
// Created by malav on 4/26/2022.
//

#ifndef ANN_MAT_HXX
#define ANN_MAT_HXX

#include "../prereqs.hxx"
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
        Mat(size_t num_r, size_t num_c, T* arr = nullptr);

        // copy constructor
        Mat(const Mat<T>& other);

        // move constructor
        Mat(Mat<T>&& other) noexcept;

        // destructor
        ~Mat() { delete[] _data; }

        //! --------------------------------------------------------------------------------------------------------

        //! ASSIGNMENT, BINARY, UNARY OPERATORS -------------------------------------------------------------------
        // copy assignment operator
        Mat<T>& operator=(const Mat<T>& rhs);

        // move assignment operator
        Mat<T>& operator=(Mat<T>&& other) noexcept ;

        // += operator
        Mat<T>& operator+=(const Mat<T>& other);

        // + operator
        Mat<T> operator+(const Mat<T>& rhs);

        // index operator
        T& operator()(size_t i, size_t j);

        // const index operator
        const T& operator()(size_t i, size_t j) const;

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
        T partial_dot(const Mat<T>& other, Dims p);

        // keep values in certain indices, and set all others to zero. indices must be sorted in increasing row, increasing column order
        void keep(Dims* indices);

        // fill matrix with a value
        void fill(T t);

        // set the rotation state of the matrix and restructure _data
        void set_rot(size_t n = 0);

        // add padding to matrix
        void padding(size_t padleft, size_t padright, size_t padtop, size_t padbottom);

        // create a Vector object out of this object
        Vector<T> flatten();

        // transpose the matrix
        Mat<T> transpose();

        // get number of rows of matrix (read only)
        size_t const& get_rows() const {return _rows;}

        // get number of cols of matrix (read only)
        size_t const& get_cols() const {return _cols;}

        // get rotation state
        size_t const& get_rot()  const {return _rot;}

        //! --------------------------------------------------------------------------------------------------------


    private:

        // helper for set_rot, rotate matrix CW by 90 degrees
        void rotate_once();

        // number of columns
        size_t _cols {0};

        // number of rows
        size_t _rows {0};

        // rotation state, 0 = no rotate; 1 = 90 degrees clockwise; 2 = 180 degrees clockwise; 3 = 270 degrees clockwise
        size_t _rot {0};

        // pointer to _data
        T* _data {nullptr};

        friend class Vector<T>;
};



#include "mat_impl.hxx"
#endif //ANN_MAT_HXX
