#include "Point.hpp"

//initialization static variable
int Point::count_tag = 0;

//constructor
Point::Point(double x, double y,double z, double h): x_coord(x) , y_coord(y) , z_coord(z) , local_mesh_ref(h){
    count_tag ++;
    tag = count_tag;
}

//every time a Point instance is created, the count tag increase, hence each point has a unique tag