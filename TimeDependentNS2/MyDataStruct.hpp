#ifndef MY_DATA_STRUCT_HPP
#define MY_DATA_STRUCT_HPP

#include <string>

struct MyDataStruct{                            //this struct contains all the data of the problem
   
   // DATA OF THE NACA
   std::string airfoil_type;                    //string that stores the name of the type of the airfoil  eg: "NACA0012"
   int last_two_digit;                          //int that stores the last two digit of the NACA type, it's used to compute the thickness of the airfoil
   double chord_length;                         //double that stores the length of the chord
   int NACA_points;                             //int that stores the number of the points that we want to use to draw half NACA profile (serve pari! vedi funzione profilo airfoil)

   // EMITTER GEOMETRY
   double radius_emitter;                       //double that stores the radius length of the emitters

   // DISTANCES
   double distance_emitter_collector;           //double that stores the distance between emitter(circle) and collector(airfoil)
   double distance_Tedge_outlet;                //double that represent the distance between the trailing edge and the end of the rectangular domain
   double distance_emitter_inlet;               //double that stores the distance between the emitter and the inlet of the domain
   double distance_emitter_up_bottom;           //double that stores the distance between the emitter and the bottom/up part of the domain

   // MESH REFINEMENT
   double mesh_ref_1;                           //double that stores corse mesh refinemet
   double mesh_ref_2;                           //double that stores less-corse mesh refinement
   double mesh_ref_3;                           //double that stores fine mesh refinement
   double mesh_ref_4;                           //double taht stores super fine mesh refinement

   // OTHER REGIONS FOR REFINEMENTS
   double cylinder_emitter_radius;              //double that stores the length of half the edge of the box that wraps the emitter
   double box_profile_semi_minor_axis;          //double that stores the length of half the semi minor edge of the box that wraps the profile
   double box_profile_semi_major_axis;          //double that stores the length of half the semi major edge of the box that wraps the profile

   // BOUNDARY LAYER
   double BL_ratio;                             //double that stores ratio between two successive layers of BL
   double BL_size;                              //double that stores the mesh size normal to the curve
   double BL_thickness;                         //double that stores the maximal thickness of the BL
   int BL_fanPoints;                            //int that stores the number of elements in the fan for each fan point(one for us)

   // ALGORITHM
   int mesh_algorithm;                          //int that stores the meshing algorithm that we want to use (see gmsh documentation)

};


//      _________________________________________________________________________________________
//      |                                                                                        |
//      |                                                                                        |
//      |                                                                                        |
//      |         _____                                                                          |
//      |        |     |                                                                         |
//      |        |_____|                                                                         |
//      |                                                                                        |
//      |                                                                                        |
//      |                                                                                        |
//      |                                                                                        |
//      |________________________________________________________________________________________|
//
//


#endif //MY_DATA_STRUCT_HPP