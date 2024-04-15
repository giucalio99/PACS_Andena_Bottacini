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
     unsigned int cutted_points;

   public:
     
     //CONSTRUCTOR
     Build_Geometry(MyDataStruct d, unsigned int c): my_data(d), cutted_points(c) {};

     //METHODS TO COMPUTE POINTS
     std::vector<Point> compute_profile(unsigned int & cut_pos) const;       //this method compute the points that made the airfoil profile, it returns in cut_pos the index in which we stop che upper part of the profile
     std::vector<Point> compute_emitter() const;       //this method compute the points that made the emitter
     std::vector<Point> compute_domain()  const;       //this method compute the points that made the rectangular domain
     void compute_trapezoid( Point & p_in ,double h, double b, double &x_coord, double &y_coord ) const ; // this method take as input the last point of the airfoil befor the cut and compute the cooordinates given heigth and half base (trapezoid shape) for the end of the TE
     
     //METHODS TO WRITE IN THE OUTPUT FILE
     void write_head(std::ofstream & ofs) const;               //this method writes in the output file the head of the .geo file
     void write_parameters(std::ofstream & ofs) const;         //this method writes in the output file the parameters that define the problem with a brief description
     void write_profile(std::ofstream & ofs) const;            //this method writes in the output file the points that compose the airfoil and the emitter
     void write_emitter(std::ofstream & ofs) const;            //this method writes in the output file the points that compose the emitter geometry
     void write_domain(std::ofstream & ofs) const;             //this method writes in the output file the points that compose the rectangular domain 
     void write_loops(std::ofstream & ofs) const;              //this method writes in the output file the loops that define the airfoil, the emitter and the domain
     void write_surface(std::ofstream & ofs) const;            //this method writes in the output file the surface that will be meshed
     void write_physical_groups(std::ofstream & ofs) const;    //this method writes in the output file the physical groups that describe the regions of the domain
     void write_BL_airfoil(std::ofstream & ofs) const;         //this method writes in the output file the boundary layer field that wraps the airfoil
     void write_BL_emitter(std::ofstream & ofs) const;         //this method wriets in the output file the boundary layer field that wraps the emitter
     void write_profile_box(std::ofstream & ofs) const;        //this method writes in the output file the field taht define the box around the airfoil
     void write_min_field(std::ofstream & ofs) const;          //this method wriets in the output file the min field
     

};

// HELPER FUNCTIONS
std::vector<double> GCL_nodes(double a, double b, int n);      //auxiliary function that compute the Gauss Cebycev Lobatto nodes
std::vector<double> Weights_mesh_ref(double a, double b, double c, std::vector<double> & x_coord);                        //auxiliary function that compute a parabolic profile that will be used to multiply that ref of the Points

#endif //BUILD_GEOMETRY_HPP