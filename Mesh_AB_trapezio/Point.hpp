#ifndef POINT_HPP
#define POINT_HPP

class Point{

  private:
    
    double x_coord;              //x-coord
    double y_coord;              //y-coord
    double z_coord;              //z-coord
    double local_mesh_ref;       //mesh refinement local to the Point
    static int count_tag;        //it counts the tags of all the Points
    int tag;                     //tag of the Point

 public:

    //CONSTRUCTOR
    Point(double x, double y,double z, double h);
  
    //GETTERS
    int get_tag() const  {return this->tag;};
    double get_x() const {return this->x_coord;};
    double get_y() const {return this->y_coord;};
    double get_z() const {return this->z_coord;};
    double get_local_mesh_ref() const {return this->local_mesh_ref;};

};













#endif //POINT_HPP