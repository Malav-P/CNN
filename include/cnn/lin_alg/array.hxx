//
// Created by Malav Patel on 12/28/22.
//

#ifndef CNN_EXAMPLE_ARRAY_HXX
#define CNN_EXAMPLE_ARRAY_HXX

using namespace std;

#include "data_types.hxx"
namespace CNN {

    /**
     * An Implementation of an N-D array class.
     *
     * This array class stores an array of elements contiguously in memory of a single data type.
     * @tparam T data-type for the array
     */
    template<typename T>
    class Array {
    public:

        /**
         * Default constructor should not exist, only defined constructors allowed to be called.
         */
        Array() = default;

        /**
         * Constructor for Array class
         *
         * @param shape the dimensions of the array, given as vector of integers. Note that negative integers in this vector
         * will lead to undefined behavior
         * @param data optionally, a pointer to the data. If null, array is zero initialized. Defaults to nullptr. Note that
         * no check is performed to ensure the dimensions of shape do not go past the allocated space of the data pointer. It is the
         * caller's responsibility to ensure this
         */
        explicit Array(const vector<int> &shape, T *data = nullptr);

        /**
         * The Destructor
         *
         * @note memory is only freed if it is allocated by the constructor (in that case owndata_ will be set to true)
         */
        ~Array() { if (owndata_) { delete[] data_; }}

        /**
         * Copy Constructor
         * @param other the 'Array' object to be copied.
         *
         * @note the copy constructor performs a deep copy of the data. I.e. changes to the original array will
         * not be reflected in the copied array!
         */
        Array(const Array &other);

        /**
         * Copy Assignment Operator
         *
         * @param other the 'Array' object to copy
         * @return a reference to the copied Array
         *
         * @note the copy assignment operator performs a deep copy of the data. I.e. changes to the original Array will
         * not be reflected in the copied Array!
         */
        Array<T> &operator=(const Array &other);

        /**
         * Move constructor
         * @param other a reference to a reference of an Array object
         */
        Array(Array &&other) noexcept;

        /**
         * Move assignment operator
         * @param other a reference to a reference of an Array object
         * @return a reference to an Array containing the moved Array
         */
        Array<T> &operator=(Array &&other) noexcept;

        /**
         * Compute the offset from position zero in linear memory required to access the element at the
         * requestd index.
         * @param indices a vector of integers containing a requested index in the N-d array.
         * @return the offset in linear memory as an integer
         *
         * @example given a 2x2 array with elements {1, 2; 3, 4}.If we request the index {1,1}, the computed offset
         * will be 3, corresponding to the last element,4, stored in the linear memory
         *
         * @warning Negative integers in the vector will lead to undefined behavior.
         * @example given an array of shape {2,3,6}. If we ask for an index {-1, 2, 3}, this will lead to undefined behavior
         *
         * @warning It is the user's responsibility to prevent out-of-bounds memory access.
         * @example given an array of shape {4, 4}. If we ask for an index {2, 5} the computed offset will be less than
         * 15 but we will not be indexing the correct value of the matrix. This will lead to undefined behavior.
         * @example given an array of shape {4, 4} If we ask for an index {10, 13} the computed offset will be greater
         * than 15, which goes beyond the allocated memory for the array. Thi swill lead to undefined behavior.
         */
        int offset(const vector<int> &indices);

        /**
         * Stride Tricks to view arrays as having a different shape, but without having to reallocate and copy over members
         *
         * @param shape a vector of non-negative integers describing the new dimensions of the array. Negative integers will
         * lead to undefined behavior
         * @param strides a vector of non-negative strides for the linear memory in the array
         * @return an Array object
         *
         * @see [Stride Tricks](https://towardsdatascience.com/advanced-numpy-master-stride-tricks-with-25-illustrated-exercises-923a9393ab20)
         *
         * @warning the returned Array will share the same data pointer as (*this). Thus, any changes made to the returned Array's
         * data will be reflected in (*this) object! Furthermore, no out-of-bounds access checking is performed. Use this
         * method with extreme caution
         */
        Array<T> as_strided(const vector<int>& shape, const vector<int>& strides);

        /**
         * Indexing Operator Overload (const version)
         * @param indices a vector of non-negative integers representing the requested index into the array
         * @return a const reference to the element in the array
         */
        const T& operator[](const vector<int> &indices) const { return data_[offset(indices)]; }

        /**
         * Indexing Operator Overload (mutable version)
         * @param indices a vector of non-negative integers representing the requested index into the array
         * @return a mutable reference to the element in the array
         */
        T& operator[](const vector<int> &indices) { return data_[offset(indices)]; }

        /**
         * Subtraction Operator Overaload
         * @param other another 'Array' object of the same data-type. The Array being subtracted from (*this)
         * @return an Array containing the elementwise difference of the two array
         *
         * @warning no check is performed to ensure the arrays are of the same dimension
         */
        Array<T> operator-(const Array<T> &other);

        /**
         * Perform element-wise division of two arrays
         * @param other the array of divisors
         * @param scalar an optional scalar by which to multiply the result by, by default set to 1
         * @return an 'Array' object containing the quotients
         *
         * @warning user must ensure other and (*this) have the the same number of elements. Otherwise this leads to
         * undefined behavior
         */
        Array<T> edivide(const Array &other, const T scalar=1);

        /**
         * Fill (*this) with a value
         * @param value the fill value
         */
        void fill(T value);

        /**
         * Resets the memory pointed to by data_ to the memory pointed to by newdata
         * @param newdata the pointer to the new data
         *
         * @note if (*this) owns the current memory, it will be freed.
         */
        void resetdata(T *newdata);

        /**
         * Function to write content from another array into a specific location in the current array
         * @param other the other 'Array' object from which data will be written
         * @param start the start position in the current 'Array' as a positive integer, indexing into linear memory
         *
         * @warning it is  the user's responsibility to ensure that the length of data written will not cause
         * out-of-bounds memory access in the current array. Otherwise this leads to undefined behavior
         */
        void write(const Array<T> &other, size_t start);

        /**
         * Reshapes the dimensions of the array
         *
         * @param newshape a vector of positive integers representing the new dimensions of the array.If the total number of elements specified by the new dimensions is not
         * equal to the current holding capacity of (*this), current memory is deallocated and the appropriate amount of new memory is
         * allocated and zero initialized.
         */
        void Reshape(const vector<int> &newshape);

        /**
         * Pads an array along the given dimensions
         *
         * @param padims a vector of integers. Positive integers means the addition of a zeros along the requested axis. Negative
         * integers means the removal of data along the requested axis.
         * @return an 'Array' object that is padded.
         *
         * @note as of this writing, pad only works with 'Array' objects with dimension 3, i.e 3D arrays. As such
         * padims will allow 6 arguments only.
         *
         * @example
         * Given an array of dimensions {2,5,5} and a padims vector {-1, 0, 2, 1, 0, -2}. This tells the function
         * to remove the first slice of data along the 1st axis, do not remove any data from the end of the 1st axis, add
         * 2 rows of zeros along the beginning of the second axis, add 1 row of zeros at the end of the second axis, do not
         * remove any columns of data along the third axis, remove 2 columns of data at the end of the third axis. Note
         * that if too many dimensions are removed along an axis, this will lead to undefined behavior, as the shape of
         * the array will become negative, which is physically non-intuitive.
         */
        Array<T> pad(const vector<int> &padims);

        /**
         * Rotate a 3D array clockwise by 90 degrees.
         *
         * This function rotates each slice of a 3D array clockwise by 90 degrees. For an image with k channels,
         * i rows, and j columns, the indices i and j are moved around during the rotation. Each channel is rotated
         * by itself
         * @return the rotated array
         */
        Array<T> rotate();

        /**
         * Print the array - TODO or remove
         */
        void print() {}

        /**
         * Utility function to return the shape of the array
         * @return a const reference to a vector of integers, the shape of array
         */
        const vector<int>& getshape() const { return shape_; }

        /**
         * Utility function to return the strides of the array
         * @return a const reference to a vector of integers, the strides of the array
         */
        const vector<int>& getstride() const {return strides_;}

        /**
         * Utility function to return the size (or equivalently, the number of elements) in the array
         * @return a const reference to an unsigned integer, the size of the array
         */
        const size_t& getsize() const {return size_;}

        /**
         * Utility function to return a mutable reference to the data held by (*this)
         * @return a reference to a mutable pointer to the data
         */
        T*& getdata() {return data_;}

        /**
         * Utility function to return a read-only reference to the data held by (*this)
         * @return a const reference to a pointer to the data
         */
        T* const& getdata() const { return data_; }


    private:

        /// dimensions of array
        vector<int> shape_{0};

        /// strides of array
        vector<int> strides_{0};

        /// size of array
        size_t size_{0};

        /// capacity of array in bytes
        size_t capacity_{0};

        /// ownership of data
        bool owndata_ {false};

        /// pointer to data
        T *data_{nullptr};

    };
}

#include "array_impl.hxx"
#endif //CNN_EXAMPLE_ARRAY_HXX
