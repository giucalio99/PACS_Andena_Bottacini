/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2021 by the deal.II authors
 *
 * This file is based on step-6 of the examples section of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 *
 * Author: Matteo Menessini, Politecnico di Milano, 2023
 *
 */

// Time-stepping from step-26

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h> // for the timer

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h> // For neighbor renumbering
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_interface_values.h> // For gradient evaluator

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h> // To use manifolds
#include <deal.II/grid/grid_in.h> // For GMSH
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h> // For UMFPACK
#include <deal.II/lac/solver_gmres.h> // For GMRES
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_ilu.h> // ILU preconditioning
#include <deal.II/lac/petsc_block_sparse_matrix.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h> // For Laplace Matrix
#include <deal.II/numerics/fe_field_function.h> // For boundary values
#include <deal.II/numerics/solution_transfer.h> // For the solution transfer
#include <deal.II/numerics/error_estimator.h> // Kelly error estimator

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_tr

#include <fstream>
#include <cmath>

#include"Problem.hpp"
#include "MyDataStruct.hpp"
#include "json.hpp"

using json = nlohmann::json;

using namespace dealii;


// Collector Manifold - START

double get_collector_height(const double &p, const MyDataStruct &s_data)
{
  
  //const double collector_length = s_data.chord_length; // [m] collector length  
  //const double g = s_data.distance_emitter_collector;

  double collector_length = 1.;
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
Point<dim> CollectorGeometry<dim>::push_forward(const Point<dim-1>  &x) const          //To not have a virtual class 
{
  Point<dim> p;

  return p;                                                                              
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
  // const std::string filename = "../../Structured_Meshes/coarse_WW.msh";
  const std::string filename = "../../Structured_Meshes/structured_naca.msh";
  cout << "Reading from " << filename << std::endl;
  std::ifstream input_file(filename);
  GridIn<2>       grid_in;
  grid_in.attach_triangulation(tria);            //Attach this triangulation to be fed with the grid data
  grid_in.read_msh(input_file);                           //Read grid data from an msh file

  const types::manifold_id emitter = 1;                   //The type used to denote manifold indicators associated with every object of the mesh

  /*const double re = s_data.radius_emitter ; // [m] emitter radius                                 
  const double g = s_data.distance_emitter_collector; // [m] interelectrode distance
  const double X = -re-g; // [m] emitter center */

  // for wire wire simulation
  // double r_col = 1e-3;
  // double r_emi = 30e-5;
  // double dist_emi_col = 0.025;
  // const double X = -r_emi-dist_emi_col;

  double X = -2.53;
  const Point<2> center(X,0.0);
  //const Point<2> center1(0.0,0.0);
  SphericalManifold<2> emitter_manifold(center);

  const types::manifold_id collector = 2;
  CollectorGeometry<2> collector_manifold; 
  // const Point<2> center2(r_col,0.0);
  // SphericalManifold<2> collector_manifold(center2);               

  tria.set_all_manifold_ids_on_boundary(1, emitter);
  tria.set_manifold(emitter, emitter_manifold);
  tria.set_all_manifold_ids_on_boundary(2, collector);
  tria.set_manifold(collector, collector_manifold);
  cout  << "Active cells: " << tria.n_active_cells() << std::endl;
}






int main()
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);  
      parallel::distributed::Triangulation<2> tria(MPI_COMM_WORLD);
      create_triangulation(tria, s_data);
      Problem<2> our_problem(tria);
      our_problem.run();
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

      return 1; // Report an error
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
