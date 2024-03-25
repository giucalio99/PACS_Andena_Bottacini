#ifndef BUILD_GEOMETRY_HPP
#define BUILD_GEOMETRY_HPP

#include "MyDataStruct.hpp"
#include "Point.hpp"
#include <vector>
#include <iostream>
#include <fstream> 
#include <string>
#include <cmath>

class Build_Geometry{

   private:

     MyDataStruct my_data;

   public:
     
     //CONSTRUCTOR
     Build_Geometry(MyDataStruct d): my_data(d) {};

     //METHODS TO COMPUTE POINTS
     std::vector<Point> compute_profile() const;       //this method compute the points that made the airfoil profile
     std::vector<Point> compute_emitter() const;       //this method compute the points that made the emitter
     std::vector<Point> compute_domain()  const;       //this method compute the points that made the rectangular domain
     
     //METHODS TO WRITE IN THE OUTPUT FILE
     void write_head(std::ofstream & ofs) const;               //this method writes in the output file the head of the .geo file
     void write_profile(std::ofstream & ofs) const;            //this method writes in the output file the points that compose the airfoil and the emitter
     void write_emitter(std::ofstream & ofs) const;            //this method writes in the output file the points that compose the emitter geometry
     void write_domain(std::ofstream & ofs) const;             //this method writes in the output file the points that compose the rectangular domain 
     void write_loops(std::ofstream & ofs) const;              //this method writes in the output file the loops that define the airfoil, the emitter and the domain
     void write_surface(std::ofstream & ofs) const;            //this method writes in the output file the surface that will be meshed
     void write_physical_groups(std::ofstream & ofs) const;    //this method writes in the output file the physical groups that describe the regions of the domain
     void write_boundary_layer(std::ofstream & ofs) const;     //this method writes in the output file the boundary layer field that wraps the airfoil
     void write_emitter_cylinder(std::ofstream & ofs) const;   //this method wriets in the output file the field that define the cylinder around the emitter
     void write_profile_box(std::ofstream & ofs) const;        //this method writes in the output file the field taht define the box around the airfoil
     void write_min_field(std::ofstream & ofs) const;          //this method wriets in the output file the min field
     void write_algorithm(std::ofstream & ofs) const;          //this method writes in the output file the meshing algorithm and the algorithm characteristics

};

//FUNCTIONS
std::vector<double> GCL_nodes(double a, double b, int n);      //auxiliary function that compute the Gauss Cebycev Lobatto nodes

#endif //BUILD_GEOMETRY_HPP