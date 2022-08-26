//
// Created by malav on 6/21/2022.
//

#ifndef ANN_PAIR_HXX
#define ANN_PAIR_HXX

template<typename T1 = unsigned int, typename T2 = unsigned int>
class Dimensions {
    public:

    Dimensions() = default;

    Dimensions(T1 in_width, T2 in_height)
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

#endif //ANN_PAIR_HXX
