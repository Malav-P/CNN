
#include "../classes/lin_alg/data_types.hxx"
#include <iostream>

int main()
{
//    int arr[8] = {4, 2, 1, 6, 8, 4, 0, 1};
//    int vec[4] = {1,2,3,4};
//
//    Mat<int> v(2, 4, arr);
//    v.padding(1, 1, 1, 1);
//
//    v.print();
//
//    v.crop(1,1,1,1);
//    v.set_rot(2);
//    v.print();

int arr[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
Cuboid<int> c(2, 2, 3, arr);
c.set_rot(2);
c.print();


}