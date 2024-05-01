/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2010 - 2023 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * Part of the source code is dual licensed under Apache-2.0 WITH
 * LLVM-exception OR LGPL-2.1-or-later. Detailed license information
 * governing the source code and code contributions can be found in
 * LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
 *
 * ------------------------------------------------------------------------
 *
 * Authors: Wolfgang Bangerth, Texas A&M University, 2009, 2010
 *          Timo Heister, University of Goettingen, 2009, 2010
 */
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h> // for the timer
#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>  // serve per usare ConditionalOStream pcout
#include <deal.II/base/index_set.h> //serve per la classe indexset
#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h> // For UMFPACK
#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/petsc_block_sparse_matrix.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparsity_tools.h> // per distribute_sparsity_pattern
//#include <deal.II/lac/matrix_out.h> // For matrix output

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h> // For Laplace Matrix
#include <deal.II/numerics/error_estimator.h>

//To parallelize
#include <deal.II/distributed/grid_refinement.h>     //tools to operate on parallel distributed triangulations
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>                //parallel distributed triangulation


#include <fstream>
#include <iostream>
#include <limits>

#include "Electrical_Constants.hpp"

// COMMENTO DI QUESTO SUB-STEP (OUR MESH) VERSO POISSON
// per i commenti vedere step40 vanilla è scritto molto bene. in questo sub-step noi non raffiniamo la mesh
// inoltre usiamo una nostra mesh presa in input. Da risultati identici allo step40 vanilla.
// è stato tolto il namespece step40 e modificato il Cmakefile.txt

namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA


using namespace dealii;

//#############################################################################################################################################

template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  virtual double value(const Point<dim> & p, const unsigned int component = 0) const override;
};


template <int dim>
double BoundaryValues<dim>::value(const Point<dim> &p, const unsigned int ) const
{
    if (p[0] <= 0.45*L) // check the x component of the point p
     return V_E*std::log(D/ni);
    else
     return V_E*std::log(-A/ni);
}

//###############################################################################################################################################

template <int dim>
class PoissonProblem
{
public:

  PoissonProblem(parallel::distributed::Triangulation<dim> &tria);

  void run(const double tolerance, const unsigned int max_iter);

private:
  
  void setup_system();
  void assemble_system();
  void solve();
  void set_boundary_values();
  void output_results(const unsigned int cycle);

  MPI_Comm mpi_communicator;

  parallel::distributed::Triangulation<dim> &triangulation;

  FE_Q<dim>       fe;
  DoFHandler<dim> dof_handler;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  AffineConstraints<double> constraints;

  PETScWrappers::MPI::SparseMatrix system_matrix;

  PETScWrappers::MPI::Vector       current_solution;  
  PETScWrappers::MPI::Vector       newton_update;     
  PETScWrappers::MPI::Vector       system_rhs;
  

  ConditionalOStream pcout;
   
  // lo step 15 si tira dietro lo sparsity pattern
};

//----------------------------------------------------------------------------------------------------------------------------------------------

template <int dim>
PoissonProblem<dim>::PoissonProblem(parallel::distributed::Triangulation<dim> &tria)
  : mpi_communicator(MPI_COMM_WORLD)
  , triangulation(tria)
  , fe(1)
  , dof_handler(tria)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
{}

//--------------------------------------------------------------------------------------------------------------------------------------------------

template <int dim>
void PoissonProblem<dim>::setup_system()
{

  dof_handler.distribute_dofs(fe);

  pcout << "   Number of active cells:       "
        << triangulation.n_global_active_cells() << std::endl
        << "   Number of degrees of freedom: " << dof_handler.n_dofs()
        << std::endl;


  // INDEX SETS INITIALIZATION
  locally_owned_dofs = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  // PETSC VECTORS 
  current_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  newton_update.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  system_rhs.reinit(locally_owned_dofs, mpi_communicator);

  // CONSTRAINTS
  constraints.clear();
  constraints.reinit(locally_relevant_dofs);
  VectorTools::interpolate_boundary_values(dof_handler, 1, Functions::ConstantFunction<dim>(V_E*std::log(D/ni)), constraints);
  VectorTools::interpolate_boundary_values(dof_handler, 2, Functions::ConstantFunction<dim>(-V_E*std::log(A/ni)), constraints);
  //DoFTools::make_hanging_node_constraints(dof_handler, constraints);  // non li abbiamo
  constraints.close();

  // DYNAMIC SPARSITY PATTERN    ( no .clear() sulla matrice? )
  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false); //noi mettavamo true
  SparsityTools::distribute_sparsity_pattern(dsp,
                                              dof_handler.locally_owned_dofs(),
                                              mpi_communicator,
                                              locally_relevant_dofs);

  system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp,  mpi_communicator);
}

//---------------------------------------------------------------------------------------------------------------------------------------------------


template <int dim>
void PoissonProblem<dim>::assemble_system()
{

  const QGauss<dim> quadrature_formula(fe.degree + 1);
  
  system_matrix = 0;
  system_rhs    = 0;

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);


  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  
  std::vector<double> old_solution(n_q_points);
  std::vector<Tensor<1, dim>> old_solution_gradients(n_q_points);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        cell_matrix = 0.;
        cell_rhs    = 0.;

        fe_values.reinit(cell);

        fe_values.get_function_gradients(current_solution, old_solution_gradients);
        fe_values.get_function_values(current_solution, old_solution);


        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            for (unsigned int i = 0; i < dofs_per_cell; ++i){
                  
                  // MATRIX A
                  for (unsigned int j = 0; j < dofs_per_cell; ++j){
                    cell_matrix(i, j) += eps_0*eps_r*fe_values.shape_grad(i, q_point) * fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point) + \
                                        (q0/V_E)*(ni*std::exp(old_solution[q_point]/V_E) + ni*std::exp(-old_solution[q_point]/V_E))*fe_values.shape_value(i, q_point) * fe_values.shape_value(j, q_point) * fe_values.JxW(q_point);
                  }             
                  
                  // RHS F
                  cell_rhs(i) +=  -eps_0*eps_r*old_solution_gradients[q_point]*fe_values.shape_grad(i,q_point)*fe_values.JxW(q_point) - \
                                  q0*(ni*std::exp(old_solution[q_point]/V_E)- ni*std::exp(-old_solution[q_point]/V_E))*fe_values.shape_value(i,q_point) * fe_values.JxW(q_point);
                                  //manca N(x)*psi
              }
          }

        cell->get_dof_indices(local_dof_indices);
        /*constraints.distribute_local_to_global(cell_matrix,
                                               cell_rhs,
                                               local_dof_indices,
                                               system_matrix,
                                               system_rhs);*/
        for (unsigned int i = 0; i < dofs_per_cell; ++i){

            for (unsigned int j = 0; j < dofs_per_cell; ++j){

              system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));

            }

            system_rhs(local_dof_indices[i]) += cell_rhs(i);
          }
      }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  std::map<types::global_dof_index, double> boundary_values;

  VectorTools::interpolate_boundary_values(dof_handler,
                                           1,
                                           Functions::ZeroFunction<dim>(),
                                           boundary_values);

  VectorTools::interpolate_boundary_values(dof_handler,
                                          2,
                                          Functions::ZeroFunction<dim>(),
                                          boundary_values);

                                          
  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     newton_update,
                                     system_rhs);
}

//------------------------------------------------------------------------------------------------------------------------------------------------

template <int dim>
void PoissonProblem<dim>::solve()
{

  PETScWrappers::MPI::Vector completely_distributed_solution(locally_owned_dofs, mpi_communicator); //temp vector

  SolverControl sc_p;     
  PETScWrappers::SparseDirectMUMPS solverMUMPS(sc_p);     
  solverMUMPS.solve(system_matrix, completely_distributed_solution, system_rhs);

  newton_update = completely_distributed_solution;

  current_solution += newton_update;

}

//--------------------------------------------------------------------------------------------------------------------------------------------

template <int dim>
void PoissonProblem<dim>::set_boundary_values()
{
  std::map<types::global_dof_index, double> boundary_values;

  VectorTools::interpolate_boundary_values(dof_handler,
                                           1,
                                           BoundaryValues<dim>(),
                                           boundary_values);

  VectorTools::interpolate_boundary_values(dof_handler,
                                          2,
                                          BoundaryValues<dim>(),
                                          boundary_values); 

  for (auto &boundary_value : boundary_values){
    current_solution(boundary_value.first) = boundary_value.second;
  }

  constraints.distribute(current_solution);
}

//-----------------------------------------------------------------------------------------------------------------------------------------------------

template <int dim>
void PoissonProblem<dim>::output_results(const unsigned int cycle)
{

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(current_solution, "phi");

  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();

  // The final step is to write this data to disk. We write up to 8 VTU files
  // in parallel with the help of MPI-IO. Additionally a PVTU record is
  // generated, which groups the written VTU files.
  data_out.write_vtu_with_pvtu_record(
    "./", "solution", cycle, mpi_communicator, 2, 8);
}


//---------------------------------------------------------------------------------------------------------------------------------------------------

template <int dim>
void PoissonProblem<dim>::run(const double tolerance, const unsigned int max_iter)
{
  
  pcout << " This is the only cycle "<< std::endl;

  setup_system();
  set_boundary_values();

  double residual_norm = std::numeric_limits<double>::max();
  
  pcout << " Initial residual: " << residual_norm << std::endl;
 
  for (unsigned int inner_iteration = 0; inner_iteration < max_iter && residual_norm > tolerance; ++inner_iteration)
    {
      assemble_system();
      solve();

      residual_norm = newton_update.linfty_norm();

      pcout << "  Residual: " << residual_norm << std::endl;
    }

  output_results(0);

  pcout << std::endl;
}

//-----------------------------------------------------------------------------------------------------------------------------------------------------
// HELPER FUNCTION FOR GRID IN 

template <int dim>
void create_triangulation(parallel::distributed::Triangulation<dim> &tria)
{
  const std::string filename = "../../../../Structured_Meshes/Structured_Square.msh";

  ConditionalOStream pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

  std::ifstream input_file(filename);
  pcout << "Reading the mesh from " << filename << std::endl;
  GridIn<2>  grid_in; //This class implements an input mechanism for grid data. It allows to read a grid structure into a triangulation object
  grid_in.attach_triangulation(tria); //we pass to grid_in our (empty) triangulation
  grid_in.read_msh(input_file); // read the msh file

}
  

//######################### MAIN #############################################################################################################


int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      parallel::distributed::Triangulation<2> tria(MPI_COMM_WORLD);
      create_triangulation(tria);

      PoissonProblem<2> poisson_problem_2d(tria);
      poisson_problem_2d.run(1e-3, 100);
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
