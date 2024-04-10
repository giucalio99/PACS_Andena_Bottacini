/* -----------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2017 Jie Cheng
 *
 * This file is part of the deal.II code gallery.
 *
 * -----------------------------------------------------------------------------
 *
 * Author: Jie Cheng <chengjiehust@gmail.com>
 */

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>

// This file includes UMFPACK: the direct solver:
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/lac/petsc_block_sparse_matrix.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/grid/grid_in.h>                //Aggiunto da me
#include <deal.II/base/geometry_info.h>          //Aggiunto da me
#include <deal.II/grid/manifold_lib.h> // To use manifolds

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <fstream>
#include <iostream>
#include <sstream>

#include "InsIMEX.hpp"
#include "MyDataStruct.hpp"
#include "json.hpp"

using json = nlohmann::json;



// // Physical Constants
// const double eps_0 = 8.854 * 1.e-12; //[F/m]= [C^2 s^2 / kg / m^3]
// const double eps_r = 1.0006;
// const double q0 = 1.602 * 1.e-19; // [C]
// const double kB = 1.381 * 1.e-23 ; //[J/K]

// const bool stratosphere = false; // if false, use atmospheric 0 km conditions

// // Input Parameters
// const double E_ON = 3.31e+6; // onset field threshold [V/m]
// const double E_ref = 7.e+7; // [V/m] maximum field value
// const double N_ref = 1.e+13; // [m^-3] maximum density value
// const double N_0 = stratosphere ? 2.2e-3 : 0.5e-3; // [m^-3] ambient ion density
// const double p_amb = stratosphere ? 5474 : 101325;
// const double T = stratosphere ? 217. : 303.; // [K] fluid temperature
// const double p_over_rho = 0.;// Boundary condition at fluid outlet
// const double delta = p_amb/101325*298/T;
// const double rho = stratosphere ? 0.089 : 1.225; // kg m^-3
// const double Mm = 2.9e-2; // kg m^-3,average air molar mass
// const double Avo = 6.022e+23; // Avogadro's number

// const double q_over_eps_0 = q0 / eps_0; // [m^3 kg C^-1 s^-2]
// const double mu0 = 1.83e-4; // [m^2/(s*V)] from Moseley
// const double mu = mu0 * delta; // scaled mobility from Moseley      
// const double V_E = kB * T / q0; // [V] ion temperature
// const double D = mu * V_E;
// //const double n_air = rho / Mm * Avo; // m^-3

// // emitter
// const double Ve = 2.e+4; // [V] emitter voltage

// // Peek's law (empyrical)
// const double eps = 1.; // wire surface roughness correction coefficient
// const double Ep = E_ON*delta*eps*(1+0.308/std::sqrt(Re*1.e+2*delta));
// const double Ri = Ep/E_ON*Re; // [m] ionization radius
// const double Vi = Ve - Ep*Re*log(Ep/E_ON); // [V] voltage on ionization region boundary



using namespace dealii;


// Collector Manifold - START

double get_collector_height(const double &p, const MyDataStruct &s_data)
{
  
  const double collector_length = s_data.chord_length; // [m] collector length  
  //const double g = s_data.distance_emitter_collector;

  //const double x = (p-g)/collector_length;
  const double x = p/collector_length;
	double y = 0;

	if ( abs(x-1.) > 1e-12 && abs(x) > 1e-12 ) {
		double a0 = 0.2969;
		double a1 = -0.126;
		double a2 = -0.3516;
		double a3 = 0.2843;
		double a4 = -0.1036; // or -0.1015 for an open trailing edge
		double t = s_data.last_two_digit; // Last 2 digits of the NACA by 100

		y = 5*t*( a0 * sqrt(x) + a1 * x + a2 * pow(x,2.0) + a3 * pow(x,3.0) + a4 * pow(x,4.0) );
	}

	return y * collector_length;
}



template <int dim>
class CollectorGeometry : public ChartManifold<dim, dim, dim-1>     //ChartManifold is a class describes mappings that can be expressed in terms of charts.
  {
public:
  virtual Point<dim-1> pull_back(const Point<dim> &space_point) const override;        //Pull back the given point in spacedim to the Euclidean chartdim dimensional space

  virtual Point<dim> push_forward(const Point<dim-1> &chart_point) const override;     //Given a point in the chartdim dimensional Euclidean space, this method returns a point on the manifold embedded in the spacedim Euclidean space.
  Point<dim> push_forward(const Point<dim-1> &chart_point, const MyDataStruct s_data) const; 
  
  virtual std::unique_ptr<Manifold<dim, dim>> clone() const override;                  //Return a copy of this manifold

  };

template <int dim>
std::unique_ptr<Manifold<dim, dim>> CollectorGeometry<dim>::clone() const
  {
return std::make_unique<CollectorGeometry<dim>>();
  }


template <int dim>
Point<dim> CollectorGeometry<dim>::push_forward(const Point<dim-1>  &x) const          //Input: a chart point that in our case is a 1D point 
{
  //const double y = get_collector_height(x[0], s_data);

  Point<dim> p;
  //p[0] = x[0]; p[1] = y;

  // if (dim == 3) {
  // p[2] = x[1];
  // }

  return p;                                                                              //Output: a point of our collector in 2D 
}

template <int dim>
Point<dim> CollectorGeometry<dim>::push_forward(const Point<dim-1>  &x, const MyDataStruct s_data) const          //Input: a chart point that in our case is a 1D point 
{
  const double y = get_collector_height(x[0], s_data);

  Point<dim> p;
  p[0] = x[0]; p[1] = y;

  if (dim == 3) {
  p[2] = x[1];
  }

  return p;                                                                              //Output: a point of our collector in 2D 
}


template <int dim>
Point<dim-1>  CollectorGeometry<dim>::pull_back(const Point<dim> &p) const             //Input: a point in our 2D mesh
{
  Point<dim-1> x;
  x[0] = p[0];

  if (dim == 3) {
  x[1] = p[2];
  }

  return x;                                                                              //Output: a chart point that in our case is a 1D point
}  
// Collector Manifold - END

 


 void create_triangulation(parallel::distributed::Triangulation<2> &tria, const MyDataStruct s_data)
{ 
  const std::string filename = "../../Meshes/REAL_EMITTER.msh";
  cout << "Reading from " << filename << std::endl;
  std::ifstream input_file(filename);
  GridIn<2>       grid_in;
  grid_in.attach_triangulation(tria);            //Attach this triangulation to be fed with the grid data
  grid_in.read_msh(input_file);                           //Read grid data from an msh file

  const types::manifold_id emitter = 1;                   //The type used to denote manifold indicators associated with every object of the mesh

  const double re = s_data.radius_emitter ; // [m] emitter radius                                 
  const double g = s_data.distance_emitter_collector; // [m] interelectrode distance
  const double X = -re-g; // [m] emitter center 
  
  const Point<2> center(X,0.0);
  SphericalManifold<2> emitter_manifold(center);

  const types::manifold_id collector = 2;
  CollectorGeometry<2> collector_manifold;                

  tria.set_all_manifold_ids_on_boundary(1, emitter);
  tria.set_manifold(emitter, emitter_manifold);
  tria.set_all_manifold_ids_on_boundary(2, collector);
  tria.set_manifold(collector, collector_manifold);
  cout  << "Active cells: " << tria.n_active_cells() << std::endl;
}




// @sect3{main function}
//
int main(int argc, char *argv[])
{
// RETRIVE DATA FROM .JSON FILE 

// Open a file stream for reading
std::ifstream inFile("Data.json");

// Check if the file stream is open
if (!inFile.is_open()) {
  std::cerr << "Failed to open the file for reading." << std::endl;
  return 1;
}

// Read JSON data from the file
json json_data;       //object type json
inFile >> json_data;

// Close the file stream
inFile.close();

// Access the data from the JSON object and store them in MyDataStruct object

MyDataStruct s_data;  //structured data

s_data.airfoil_type = json_data["airfoil_type"];       
s_data.last_two_digit = json_data["last_2_digit_NACA"];           
s_data.chord_length = json_data["chord_length"];            
s_data.radius_emitter = json_data["radius_emitter"];
s_data.distance_emitter_collector=json_data["distance_emitter_collector"];
s_data.distance_Tedge_outlet=json_data["distance_Tedge_outlet"];
s_data.distance_emitter_inlet=json_data["distance_emitter_inlet"];
s_data.distance_emitter_up_bottom=json_data["distance_emitter_up_bottom"];
s_data.cylinder_emitter_radius=json_data["cylinder_emitter_radius"];
s_data.box_profile_semi_minor_axis=json_data["box_profile_semi_minor_axis"];
s_data.box_profile_semi_major_axis=json_data["box_profile_semi_major_axis"];



try
  {
    using namespace dealii;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 2);            //Initialize MPI (and, if deal.II was configured to use it, PETSc) and set the maximum number of threads used by deal.II to the given parameter.
    parallel::distributed::Triangulation<2> tria(MPI_COMM_WORLD);
    create_triangulation(tria, s_data);
    InsIMEX<2> flow(tria);
    flow.run();
  }
catch (std::exception &exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
return 0;
}
