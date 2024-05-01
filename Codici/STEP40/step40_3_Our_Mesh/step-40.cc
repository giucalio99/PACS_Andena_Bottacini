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
#include <cmath>
#include <iostream>


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


template <int dim>
class LaplaceProblem
{
public:

  LaplaceProblem(parallel::distributed::Triangulation<dim> &tria);

  void run();

private:
  
  void setup_system();
  void assemble_system();
  void solve();
  void output_results(const unsigned int cycle);

  MPI_Comm mpi_communicator;

  parallel::distributed::Triangulation<dim> &triangulation;

  FE_Q<dim>       fe;
  DoFHandler<dim> dof_handler;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  AffineConstraints<double> constraints;

  PETScWrappers::MPI::SparseMatrix system_matrix;
  PETScWrappers::MPI::Vector       locally_relevant_solution;
  PETScWrappers::MPI::Vector       system_rhs;

  ConditionalOStream pcout;
  TimerOutput        computing_timer;
};



template <int dim>
LaplaceProblem<dim>::LaplaceProblem(parallel::distributed::Triangulation<dim> &tria)
  : mpi_communicator(MPI_COMM_WORLD)
  , triangulation(tria)
  , fe(2)
  , dof_handler(tria)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  , computing_timer(mpi_communicator, pcout, TimerOutput::never, TimerOutput::wall_times)
{}




template <int dim>
void LaplaceProblem<dim>::setup_system()
{
  TimerOutput::Scope t(computing_timer, "setup");

  dof_handler.distribute_dofs(fe);

  pcout << "   Number of active cells:       "
        << triangulation.n_global_active_cells() << std::endl
        << "   Number of degrees of freedom: " << dof_handler.n_dofs()
        << std::endl;



  locally_owned_dofs = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);


  locally_relevant_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  system_rhs.reinit(locally_owned_dofs, mpi_communicator);

  


  constraints.clear();
  constraints.reinit(locally_relevant_dofs);
  //DoFTools::make_hanging_node_constraints(dof_handler, constraints);  // non li abbiamo
  VectorTools::interpolate_boundary_values(dof_handler, 0, Functions::ZeroFunction<dim>(), constraints);
  constraints.close();


  DynamicSparsityPattern dsp(locally_relevant_dofs);

  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                              dof_handler.locally_owned_dofs(),
                                              mpi_communicator,
                                              locally_relevant_dofs);

  system_matrix.reinit(locally_owned_dofs,
                        locally_owned_dofs,
                        dsp,
                        mpi_communicator);
}




template <int dim>
void LaplaceProblem<dim>::assemble_system()
{
  TimerOutput::Scope t(computing_timer, "assembly");

  const QGauss<dim> quadrature_formula(fe.degree + 1);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        cell_matrix = 0.;
        cell_rhs    = 0.;

        fe_values.reinit(cell);

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            const double rhs_value =
              (fe_values.quadrature_point(q_point)[1] >
                    0.5 +
                      0.25 * std::sin(4.0 * numbers::PI *
                                      fe_values.quadrature_point(q_point)[0]) ?
                  1. :
                  -1.);

            for (unsigned int i = 0; i < dofs_per_cell; ++i){

                  for (unsigned int j = 0; j < dofs_per_cell; ++j){
                    cell_matrix(i, j) += fe_values.shape_grad(i, q_point) *
                                          fe_values.shape_grad(j, q_point) *
                                          fe_values.JxW(q_point);
                  }

                  cell_rhs(i) += rhs_value *                         
                                  fe_values.shape_value(i, q_point) * 
                                  fe_values.JxW(q_point);
              }
          }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix,
                                                cell_rhs,
                                                local_dof_indices,
                                                system_matrix,
                                                system_rhs);
      }


  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}


template <int dim>
void LaplaceProblem<dim>::solve()
{
  TimerOutput::Scope t(computing_timer, "solve");

  PETScWrappers::MPI::Vector completely_distributed_solution(locally_owned_dofs, mpi_communicator);

  SolverControl solver_control(dof_handler.n_dofs(), 1e-12);

  PETScWrappers::SolverCG  solver(solver_control);


  LA::MPI::PreconditionAMG::AdditionalData data;
#ifdef USE_PETSC_LA
  data.symmetric_operator = true;
#else
  /* Trilinos defaults are good */
#endif
  LA::MPI::PreconditionAMG preconditioner;
  preconditioner.initialize(system_matrix, data);

  solver.solve(system_matrix,
                completely_distributed_solution,
                system_rhs,
                preconditioner);

  pcout << "   Solved in " << solver_control.last_step() << " iterations."
        << std::endl;

  constraints.distribute(completely_distributed_solution);

  locally_relevant_solution = completely_distributed_solution;
}



template <int dim>
void LaplaceProblem<dim>::output_results(const unsigned int cycle)
{
  TimerOutput::Scope t(computing_timer, "output");

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(locally_relevant_solution, "u");

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




template <int dim>
void LaplaceProblem<dim>::run()
{
  pcout << "Running with "
#ifdef USE_PETSC_LA
        << "PETSc"
#else
        << "Trilinos"
#endif
        << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
        << " MPI rank(s)..." << std::endl;

  
      pcout << " This is the only cycle "<< std::endl;

      setup_system();
      assemble_system();
      solve();
      output_results(0);

      computing_timer.print_summary();
      computing_timer.reset();

      pcout << std::endl;
}


// HELPER FUNCTION FOR GRID IN 

template <int dim>
void create_triangulation(parallel::distributed::Triangulation<dim> &tria)
{
  const std::string filename = "../../../../Structured_Meshes/Structured_step40.msh";

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

      LaplaceProblem<2> laplace_problem_2d(tria);
      laplace_problem_2d.run();
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
