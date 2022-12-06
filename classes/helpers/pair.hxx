//
// Created by malav on 6/21/2022.
//

#ifndef ANN_PAIR_HXX
#define ANN_PAIR_HXX

namespace CNN {

    template<typename T1 = size_t, typename T2 = size_t>
    class Dimensions {
    public:

        Dimensions() = default;

        Dimensions(T1 in_width, T2 in_height)
                : width(in_width),
                  height(in_height) {}

        friend bool operator==(const Dimensions &l, const Dimensions &r) {
            if (l.width == r.width && l.height == r.height) { return true; }
            else { return false; }
        }

        T1 width{0};

        T2 height{0};
    };

    template<typename T1 = size_t, typename T2 = size_t, typename T3 = size_t>
    class Dimensions3 {
    public:

        Dimensions3() = default;

        Dimensions3(T1 in_width, T2 in_height, T3 in_depth)
                : width(in_width),
                  height(in_height),
                  depth(in_depth) {}

        friend bool operator==(const Dimensions3 &l, const Dimensions3 &r) {
            if (l.width == r.width && l.height == r.height && l.depth == r.depth) { return true; }
            else { return false; }
        }

        T1 width{0};

        T2 height{0};

        T3 depth{0};
    };

    template<typename T1 = size_t, typename T2 = size_t, typename T3 = size_t, typename T4 = size_t>
    class Dimensions4 {
    public:

        Dimensions4() = default;

        Dimensions4(T1 first, T2 second, T3 third, T4 fourth)
                : first(first),
                  second(second),
                  third(third),
                  fourth(fourth) {}

        friend bool operator==(const Dimensions4 &l, const Dimensions4 &r) {
            if (l.first == r.first && l.second == r.second && l.third == r.third &&
                l.fourth == r.fourth) { return true; }
            else { return false; }
        }

        T1 first{0};

        T2 second{0};

        T3 third{0};

        T4 fourth{0};
    };

    using Pair = std::pair<size_t, double>;
    using Dims = Dimensions<>;
    using Dims3 = Dimensions3<>;
}



#endif //ANN_PAIR_HXX
