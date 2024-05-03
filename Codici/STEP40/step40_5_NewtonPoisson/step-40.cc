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

#include "Electrical_Constants.hpp"    // FUNZIONA ANCHE SE NON E NEL CMAKEFILE TXT ??

// COMMENTO DI QUESTO QUARTO SUB-STEP (OUR MESH) VERSO NEWTON POISSON
// per i commenti vedere step40 vanilla è scritto molto bene. in questo sub-step noi non raffiniamo la mesh
// inoltre usiamo una nostra mesh presa in input. Performiamo una iterazione di newton

using namespace dealii;


template <int dim>
class PoissonProblem
{
public:

  PoissonProblem(parallel::distributed::Triangulation<dim> &tria);

  void run(unsigned int max_iter, double toll);

private:
  
  void setup_system();
  void initialize_current_solution();
  void assemble_system();
  void solve();
  void output_results(const unsigned int cycle);


  MPI_Comm mpi_communicator;

  parallel::distributed::Triangulation<dim> &triangulation;

  FE_Q<dim>       fe;
  DoFHandler<dim> dof_handler;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  //AffineConstraints<double> constraints;       non penso che servano, in effetti se la commenti in step40-4 stessa soluzione con e senza
  AffineConstraints<double> zero_constraints;

  PETScWrappers::MPI::SparseMatrix system_matrix;

  PETScWrappers::MPI::Vector       current_solution;  
  PETScWrappers::MPI::Vector       newton_update;     
  PETScWrappers::MPI::Vector       system_rhs;

  ConditionalOStream pcout;

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
  current_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);  //ghosted
  newton_update.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);     //ghosted
  system_rhs.reinit(locally_owned_dofs, mpi_communicator);                               //non ghosted
 
  // CONSTRAINTS 
  /*
  constraints.clear();
  constraints.reinit(locally_relevant_dofs);    // SERVONO VERAMNETE LORO ?? i constraints normali, non nulli intendo
  VectorTools::interpolate_boundary_values(dof_handler, 1, Functions::ConstantFunction<dim>(V_E*std::log(D/ni)), constraints);
  VectorTools::interpolate_boundary_values(dof_handler, 2, Functions::ConstantFunction<dim>(-V_E*std::log(A/ni)), constraints);
  */
  // ZERO CONSTRAINTS
  zero_constraints.clear();
  zero_constraints.reinit(locally_relevant_dofs);
  VectorTools::interpolate_boundary_values(dof_handler, 1, Functions::ZeroFunction<dim>(), zero_constraints);
  VectorTools::interpolate_boundary_values(dof_handler, 2, Functions::ZeroFunction<dim>(), zero_constraints);
  zero_constraints.close();

  //DoFTools::make_hanging_node_constraints(dof_handler, constraints);  // non li abbiamo

  // DYNAMIC SPARSITY PATTERN    
  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp, zero_constraints, false); //noi mettavamo true;  prima era constraints e basta
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             dof_handler.locally_owned_dofs(),
                                             mpi_communicator,
                                             locally_relevant_dofs);
  
  system_matrix.clear();
  system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp,  mpi_communicator);
  
  pcout << " End of setup_system "<< std::endl<<std::endl;

}

//---------------------------------------------------------------------------------------------------------------------------------------------------

template <int dim>
void PoissonProblem<dim>:: initialize_current_solution(){

  //NB: the L2 norm of current solution without constraints is zero, correct (without = 0)
  //NB: abbiamo constatato che setup_system scrive effettivamente in constraints i valori imposti
  //    ergo in constrints ho effettivamente i valori delle BC legate ai vari dof dei processori VEDERE CARTELLA OUTPUT AFFINE CONST
  //NB: finalmente in current solution ci sono dei valori sensati, una volta impostate le BCs ha norma L2 non banale e 
  //    norma L_INF coerente con i valori delle BCs.  QUESTA FUNCTION FA IL SUO
  //    unica cosa brutta l'uso di un temp vector
 
  PETScWrappers::MPI::Vector temp;
	temp.reinit(locally_owned_dofs, mpi_communicator);   //non ghosted, serve per imporre i valori delle BCs
	temp = current_solution; //current solution here is zero by default constructor
	  
  std::map<types::global_dof_index, double> boundary_values;
  
  VectorTools::interpolate_boundary_values(dof_handler,
                                           1,
                                           Functions::ConstantFunction<dim>(V_E*std::log(D/ni)),
                                           boundary_values);

  VectorTools::interpolate_boundary_values(dof_handler,
                                           2,
                                           Functions::ConstantFunction<dim>(-V_E*std::log(A/ni)),
                                           boundary_values);
  
  for (auto &boundary_value : boundary_values){
    temp(boundary_value.first) = boundary_value.second;
  }
  
  temp.compress(VectorOperation::insert); //giusto insert, add non funziona
  current_solution = temp;
  
  pcout << " The L2 norm of current solution is: "<< current_solution.l2_norm()<< std::endl;
  pcout << " The L_INF norm of current solution is: "<< current_solution.linfty_norm()<< std::endl;
  
  pcout << " End of initialization_current_solution "<< std::endl<<std::endl;

}

//---------------------------------------------------------------------------------------------------------------------------------------------------

// ATTENZIONE IN TEORIA QUA C'E' QUALCOSA CHE NON VA DAL PUNTO DI VISTA FISICO, VEDERE CONTI , VEDERE CELL RHS

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

            const double doping = (fe_values.quadrature_point(q_point)[0] < L/2 ? D : -A);  // value of the doping
            
            for (unsigned int i = 0; i < dofs_per_cell; ++i){
                  
                  // MATRIX A
                  for (unsigned int j = 0; j < dofs_per_cell; ++j){
                    cell_matrix(i, j) += eps_0*eps_r*fe_values.shape_grad(i, q_point) * fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point) + \
                                        (q0/V_E)*(ni*std::exp(old_solution[q_point]/V_E) + ni*std::exp(-old_solution[q_point]/V_E))*fe_values.shape_value(i, q_point) * fe_values.shape_value(j, q_point) * fe_values.JxW(q_point);
                  }             
                  
                  // RHS F
                  cell_rhs(i) +=  -eps_0*eps_r*old_solution_gradients[q_point]*fe_values.shape_grad(i,q_point)*fe_values.JxW(q_point) - \
                                  q0*(ni*std::exp(old_solution[q_point]/V_E)- ni*std::exp(-old_solution[q_point]/V_E))*fe_values.shape_value(i,q_point) * fe_values.JxW(q_point)+\
                                  q0*doping*fe_values.shape_value(i,q_point)*fe_values.JxW(q_point); //qo*N(x) integrato
                  
              }
          }

        cell->get_dof_indices(local_dof_indices);
        zero_constraints.distribute_local_to_global(cell_matrix,
                                               cell_rhs,
                                               local_dof_indices,
                                               system_matrix,
                                               system_rhs);
        
      }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
  
  pcout << " The L_INF norm of the assembled matrix is "<<system_matrix.linfty_norm() <<std::endl;
  pcout << " The L_FROB norm of the assembled matrix is "<<system_matrix.frobenius_norm() <<std::endl<<std::endl;

  pcout << " The L2 norm of the assembled rhs is "<<system_rhs.l2_norm() <<std::endl;
  pcout << " The L_INF norm of the assembled rhs is "<<system_rhs.linfty_norm() <<std::endl;

  pcout << " End of assemble_system "<< std::endl<<std::endl;
}

//------------------------------------------------------------------------------------------------------------------------------------------------

template <int dim>
void PoissonProblem<dim>::solve()
{
  // NB: Assemble funziona e riempie con elementi dell'ordine di 1e-10 sia la matrice che il rhs del system
  // NB: Solve sembrerebbe funzionare, non sono sicuro su quali setting mettere e quale solver usare,
  //     ciononostante otteniamo una current solution con una normal L2 diversa( maggiore) di quella dopo l'inizializzazione (L_INF norm uguale)
  
  double alpha = 1.0; // provare con 0.8 se non converge

  SolverControl sc_p(dof_handler.n_dofs(), 1e-10);     
  PETScWrappers::SparseDirectMUMPS solverMUMPS(sc_p);     
  solverMUMPS.solve(system_matrix, newton_update, system_rhs);
  
  PETScWrappers::MPI::Vector temp(locally_owned_dofs, mpi_communicator);
  temp = current_solution;
  temp += newton_update;
  current_solution = temp;

  pcout << "L2 norm of the current solution: " << current_solution.l2_norm() << std::endl;
  pcout << "L_INF norm of the current solution: " << current_solution.linfty_norm() << std::endl;

  pcout << " End of solve "<< std::endl<<std::endl;
  
}


//-----------------------------------------------------------------------------------------------------------------------------------------------------

template <int dim>
void PoissonProblem<dim>::output_results(const unsigned int cycle)
{

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(current_solution, "phi");
  data_out.add_data_vector(newton_update, "delta_phi");

  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();

  // The final step is to write this data to disk. We write up to 8 VTU files
  // in parallel with the help of MPI-IO. Additionally a PVTU record is
  // generated, which groups the written VTU files.
  data_out.write_vtu_with_pvtu_record(
    "./", "solution", cycle, mpi_communicator, 2, 1);

  pcout << " End of output_results"<< std::endl<<std::endl;

}


//---------------------------------------------------------------------------------------------------------------------------------------------------

template <int dim>
void PoissonProblem<dim>::run(unsigned int max_iter, double toll) // tolerance linked to the magnitude of the newton update
{
  
  unsigned int counter = 0;

  pcout << " -- START NEWTON METHOD --"<< std::endl<<std::endl<<std::endl;

  pcout << "  SETUP SYSTEM "<< std::endl;
  setup_system();
  
  pcout << "  INITIALIZATION "<< std::endl;
  initialize_current_solution();

  double residual_norm = std::numeric_limits<double>::max();

  pcout << " Initial residual: " << residual_norm << std::endl<<std::endl;
  

  while(counter < max_iter && residual_norm > toll){

    pcout << " NEWTON ITERATION NUMBER: "<< counter +1<<std::endl<<std::endl;
    pcout << " Assemble System "<< std::endl;
    assemble_system();
    pcout << " Solve System"<< std::endl;
    solve();
    residual_norm = newton_update.l2_norm();
    pcout << " Update Residual: "<<residual_norm<<std::endl<<std::endl;
    counter ++;

  }

  pcout << "  OUTPUT RESULT "<< std::endl;

  output_results(0);
  
  pcout << " -- END NEWTON METHOD -- "<< std::endl;
  
}

//-----------------------------------------------------------------------------------------------------------------------------------------------------
// HELPER FUNCTION FOR GRID IN 

template <int dim>
void create_triangulation(parallel::distributed::Triangulation<dim> &tria)
{
  const std::string filename = "../../../../Structured_Meshes/Structured_Square.msh";

  ConditionalOStream pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

  std::ifstream input_file(filename);
  pcout <<"Reading the mesh from " << filename << std::endl;
  GridIn<2>  grid_in; //This class implements an input mechanism for grid data. It allows to read a grid structure into a triangulation object
  grid_in.attach_triangulation(tria); //we pass to grid_in our (empty) triangulation
  grid_in.read_msh(input_file); // read the msh file
  pcout << " Grid read correctly "<< std::endl;

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
      poisson_problem_2d.run(1000, 1e-5);
    
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
