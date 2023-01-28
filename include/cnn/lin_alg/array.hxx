//
// Created by Malav Patel on 12/28/22.
//

#ifndef CNN_EXAMPLE_ARRAY_HXX
#define CNN_EXAMPLE_ARRAY_HXX


#include "data_types.hxx"
namespace CNN {

    using namespace std;

    template<typename T>
    class Array {
    public:

        // default constructor should not exist
        Array() = default;

        // constructor
        explicit Array(const vector<int> &shape, T *data = nullptr);

        // destructor
        ~Array() { if (owndata_) { delete[] data_; }}

        // copy constructor -- TOTEST
        Array(const Array &other);

        // copy assignment operator -- TOTEST
        Array<T> &operator=(const Array &other);

        // move constructor -- TOTEST
        Array(Array &&other) noexcept;

        // move assignment operator -- TOTEST
        Array<T> &operator=(Array &&other) noexcept;

        // compute offset from indices
        int offset(const vector<int> &indices);

        // stride tricks
        Array<T> as_strided(const vector<int>& shape, const vector<int>& strides);

        // const reference to element
        const T &operator[](const vector<int> &indices) const { return data_[offset(indices)]; }

        // mutable reference to element
        T &operator[](const vector<int> &indices) { return data_[offset(indices)]; }

        // scale array
        void operator*=(T alpha);

        // subtraction operator
        Array<T> operator-(const Array<T> &other);

        // edivide two arrays
        Array<T> edivide(const Array &other, T scalar);

        // fill array with value
        void fill(T value);

        // reset data
        void resetdata(T *newdata);

        // write into array
        void write(const Array<T> &other, size_t start);

        // 2D convolution of two 2D-arrays -- TODO
        friend Array<T> conv2(const Array &A, const Array &B);

        // reshape the array
        void Reshape(const vector<int> &newshape);

        Array<T> operator*(const Array<T> &B);

        // partial dot product
        T partial_dot(Array<T> &other, const vector<int> &startpos);

        // pad the array along given dimensions with zeros
        Array<T> pad(const vector<int> &padims);

        // rotate matrix by 90 degrees clockwise
        Array<T> rotate();

        // print array
        void print() {}

        // get shape
        const vector<int> &getshape() const { return shape_; }
        const size_t& getsize() const {return size_;}
        T* getdata() {return data_;}
        // get the data (read only)
        T *const &getdata() const { return data_; }

        // get strides

        const vector<int>& getstride() const {return strides_;}


    private:

        // dimensions of array
        vector<int> shape_{0};

        // strides of array
        vector<int> strides_{0};

        // size of array
        size_t size_{0};

        // capacity of array in bytes
        size_t capacity_{0};

        // ownership of data
        bool owndata_;

        // pointer to data
        T *data_{nullptr};

        // helper function for iterating through arrays
        template<typename Lambda>
        void recurse(vector<int> &index, int i, Lambda &&func, const vector<int> &start, const vector<int> &end) {
            if (i == index.size()) {
                //cout << index[0] << index[1] << index[2] << "\n";
                func(index);
            } else {
                for (index[i] = start[i]; index[i] < end[i]; index[i]++) { recurse(index, i + 1, func, start, end); }
            }
        }

    };
}

#include "array_impl.hxx"
#endif //CNN_EXAMPLE_ARRAY_HXX
