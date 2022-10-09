//
// Created by malav on 6/21/2022.
//

#ifndef ANN_PAIR_HXX
#define ANN_PAIR_HXX

//void *operator new(size_t len) {
//    void *ptr;
//    cudaMallocManaged(&ptr, len);
//    cudaDeviceSynchronize();
//    return ptr;
//}
//
//void operator delete(void *ptr) {
//    cudaDeviceSynchronize();
//    cudaFree(ptr);}

template<typename T1 = size_t , typename T2 = size_t>
class Dimensions  {
    public:

    Dimensions() = default;

    __host__ __device__ Dimensions(T1 in_width, T2 in_height)
    : width(in_width),
      height(in_height)
    {}

    friend bool operator==(const Dimensions& l, const Dimensions& r)
    {
        if (l.width == r.width && l.height == r.height) {return true;}
        else {return false;}
    }

    T1 width {0};

    T2 height {0};
};

template<typename T1 = size_t , typename T2 = size_t, typename T3 = size_t>
class Dimensions3{
public:

    Dimensions3() = default;

    __host__ __device__ Dimensions3(T1 in_width, T2 in_height, T3 in_depth)
            : width(in_width),
              height(in_height),
              depth(in_depth)
    {}

    friend bool operator==(const Dimensions3& l, const Dimensions3& r)
    {
        if (l.width == r.width && l.height == r.height && l.depth == r.depth) {return true;}
        else {return false;}
    }

    T1 width {0};

    T2 height {0};

    T3 depth {0};
};

#endif //ANN_PAIR_HXX
