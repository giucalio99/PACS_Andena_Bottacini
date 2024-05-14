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
#include "Electrical_Values.hpp"

// COMMENTO DI QUESTO QUINTO SUB-STEP (OUR MESH) VERSO NEWTON POISSON
// per i commenti vedere step40 vanilla è scritto molto bene. in questo sub-step noi non raffiniamo la mesh
// inoltre usiamo una nostra mesh presa in input. Performiamo tante iterazioni di newton e risolviamo
// DriftDiffusion; questo è l'ultimo step

using namespace dealii;

// HELPER FUNCTIONS TO BUILD DRIFT DIFFUSION MATRIX

void bernoulli (double x, double &bp, double &bn)
{
  const double xlim = 1.0e-2;
  double ax  = fabs(x);

  bp  = 0.0;
  bn  = 0.0;

  //  X=0
  if (x == 0.0)
    {
      bp = 1.0;
      bn = 1.0;
      return;
    }

  // ASYMPTOTICS
  if (ax > 80.0)
    {
      if (x > 0.0)
        {
          bp = 0.0;
          bn = x;
        }
      else
        {
          bp = -x;
          bn = 0.0;
        }
      return;
    }

  // INTERMEDIATE VALUES
  if (ax <= 80 &&  ax > xlim)
    {
      bp = x / (exp (x) - 1.0);
      bn = x + bp;
      return;
    }

  // SMALL VALUES
  if (ax <= xlim &&  ax != 0.0)
    {
      double jj = 1.0;
      double fp = 1.0;
      double fn = 1.0;
      double df = 1.0;
      double segno = 1.0;
      while (fabs (df) > 1.0e-16)
        {
          jj += 1.0;
          segno = -segno;
          df = df * x / jj;
          fp = fp + df;
          fn = fn + segno * df;
        }
      bp = 1 / fp;
      bn = 1 / fn;
      return;
    }

};

double side_length (const Point<2> a, const Point<2> b)
{
	double length = 0.;

	if (a[0] == b[0])
		length = std::abs(a[1] - b[1]);
	else if (a[1] == b[1])
		length = std::abs(a[0] - b[0]);
	else
		length = std::sqrt(a[0]*a[0] + b[0]*b[0] - 2.*a[0]*b[0] + a[1]*a[1] + b[1]*b[1] - 2.*a[1]*b[1]);

	return length;
}

double triangle_denom(const Point<2> a, const Point<2> b, const Point<2> c)
{
	const double x1 = a[0];
	const double y1 = a[1];

	const double x2 = b[0];
	const double y2 = b[1];

	const double x3 = c[0];
	const double y3 = c[1];

	const double denom = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2);

	return denom;
}

Tensor<1,2> face_normal(const Point<2> a, const Point<2> b) {

	Tensor<1,2> tangent, normal;

	tangent[0] = b[0] - a[0];
	tangent[1] = b[1] - a[1];

	normal[0] = -tangent[1];
	normal[1] = tangent[0];

	return normal;
}


FullMatrix<double> compute_triangle_matrix(const Point<2> a, const Point<2> b, const Point<2> c, const double alpha12, const double alpha23, const double alpha31, const double D)
{
	const unsigned int size = 3;
	FullMatrix<double> tria_matrix(size,size);

	tria_matrix = 0;

	const double denom = triangle_denom(a,b,c);
	const double area = 0.5*std::abs(denom);

	const Tensor<1,2> grad_psi_1 = face_normal(b,c)/denom;
	const Tensor<1,2> grad_psi_2 = face_normal(c,a)/denom;
	const Tensor<1,2> grad_psi_3 = face_normal(a,b)/denom;

	const double l_12 = grad_psi_1 * grad_psi_2;
	const double l_23 = grad_psi_2 * grad_psi_3;
	const double l_31 = grad_psi_1 * grad_psi_3;

	double bp12, bn12, bp23, bn23, bp31, bn31;

	bernoulli(alpha12,bp12,bn12);
	bernoulli(alpha23,bp23,bn23);
	bernoulli(alpha31,bp31,bn31);

	tria_matrix(0,1) = D * area * bp12 * l_12;
	tria_matrix(0,2) = D * area * bn31 * l_31;

	tria_matrix(1,0) = D * area * bn12 * l_12;
	tria_matrix(1,2) = D * area * bp23 * l_23;

	tria_matrix(2,0) = D * area * bp31 * l_31;
	tria_matrix(2,1) = D * area * bn23 * l_23;
	
	tria_matrix(0,0) = - (tria_matrix(1,0)+tria_matrix(2,0));
	tria_matrix(1,1) = - (tria_matrix(0,1)+tria_matrix(2,1));
	tria_matrix(2,2) = - (tria_matrix(0,2)+tria_matrix(1,2));

	return tria_matrix;
}

//-------------- CLASS POISSON PROBLEM CHE DOVREBBE ESSERE CHIAMATA DRIFT DIFFUSION ----------------------------------------------------------

template <int dim>
class PoissonProblem
{
public:

  PoissonProblem(parallel::distributed::Triangulation<dim> &tria);

  void run(); // we pass to run the tolerance and the max number of iterations for newton

private:
  
  void setup_system(); // sia newton poisson che drift diffusion

  // void initialize_current_solution();
  void initialization();
  //void compute_densities();

  void assemble_laplace_matrix(bool use_nonzero_constraints);
  void assemble_mass_matrix(bool use_nonzero_constraints);
  void assemble_nonlinear_poisson(bool use_nonzero_constraints);
  void assemble_drift_diffusion_matrix();
  void one_cycle_newton_poisson(bool use_nonzero_constraints,
                                const unsigned int max_iter_newton, 
                                const double toll_newton);

  void solve_poisson(bool use_nonzero_constraints);
  void apply_drift_diffusion_boundary_conditions();
  void solve_drift_diffusion();
  

  void output_results(const unsigned int cycle);


  MPI_Comm mpi_communicator;

  parallel::distributed::Triangulation<dim> &triangulation;

  FE_Q<dim>       fe;
  DoFHandler<dim> dof_handler;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  AffineConstraints<double> constraints;       
  AffineConstraints<double> zero_constraints;
  
  AffineConstraints<double> density_constraints; 
  
  // poisson matrices
  PETScWrappers::MPI::SparseMatrix system_matrix_poisson;
  PETScWrappers::MPI::SparseMatrix laplace_matrix;
  PETScWrappers::MPI::SparseMatrix mass_matrix;
  PETScWrappers::MPI::SparseMatrix density_matrix;
  
  // drift diffusion matrices
  PETScWrappers::MPI::SparseMatrix hole_drift_diffusion_matrix;
  PETScWrappers::MPI::SparseMatrix electron_drift_diffusion_matrix;
  PETScWrappers::MPI::SparseMatrix electron_matrix;
  PETScWrappers::MPI::SparseMatrix hole_matrix;


  PETScWrappers::MPI::Vector       current_solution;  
  PETScWrappers::MPI::Vector       newton_update;     
  PETScWrappers::MPI::Vector       poisson_system_rhs;

  PETScWrappers::MPI::Vector       electron_density;
  PETScWrappers::MPI::Vector       hole_density;

  PETScWrappers::MPI::Vector       old_electron_density;
  PETScWrappers::MPI::Vector       old_hole_density;

  PETScWrappers::MPI::Vector       rhs_hole_density;
  PETScWrappers::MPI::Vector       rhs_electron_density;

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

  // PETSC VECTORS DECLARATIONS AND REINIT
  current_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);  //ghosted
  newton_update.reinit(locally_owned_dofs, mpi_communicator);                            //non ghosted
  poisson_system_rhs.reinit(locally_owned_dofs, mpi_communicator);                       //non ghosted

  electron_density.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);  // ghosted
  hole_density.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);      // ghosted

  old_electron_density.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);  // ghosted
  old_hole_density.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);      // ghosted

  rhs_electron_density.reinit(locally_owned_dofs, mpi_communicator);  // non ghosted
  rhs_hole_density.reinit(locally_owned_dofs, mpi_communicator);      // non ghosted
 
  // CONSTRAINTS FOR THE POTENTIAL SOLUTION  
  constraints.clear();
  constraints.reinit(locally_relevant_dofs);    // SERVONO VERAMNETE LORO ?? i constraints normali, non nulli intendo
  // VectorTools::interpolate_boundary_values(dof_handler, 1, Functions::ConstantFunction<dim>(V_TH*std::log(D/ni)), constraints);
  // VectorTools::interpolate_boundary_values(dof_handler, 2, Functions::ConstantFunction<dim>(-V_TH*std::log(A/ni)), constraints);
  VectorTools::interpolate_boundary_values(dof_handler, 1, Functions::ConstantFunction<dim>(0), constraints);
  VectorTools::interpolate_boundary_values(dof_handler, 2, Functions::ConstantFunction<dim>(0), constraints);
  constraints.close();
  
  // ZERO CONSTRAINTS FOR NEWTON POISSON
  zero_constraints.clear();
  zero_constraints.reinit(locally_relevant_dofs);
  VectorTools::interpolate_boundary_values(dof_handler, 1, Functions::ZeroFunction<dim>(), zero_constraints);
  VectorTools::interpolate_boundary_values(dof_handler, 2, Functions::ZeroFunction<dim>(), zero_constraints);
  // VectorTools::interpolate_boundary_values(dof_handler, 1, Functions::ConstantFunction<dim>(-0.001), zero_constraints);
  // VectorTools::interpolate_boundary_values(dof_handler, 2, Functions::ConstantFunction<dim>(-0.001), zero_constraints);
  zero_constraints.close();

  // CONSTRAINTS FOR DRIFT DIFFUSION SYSTEM
  density_constraints.clear();
  density_constraints.close();

  // DYNAMIC SPARSITY PATTERN POISSON PROBLEM AND RELATIVE MATRICES
  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp, zero_constraints, false); //noi mettavamo true;  prima era constraints e basta

  SparsityTools::distribute_sparsity_pattern(dsp,
                                             dof_handler.locally_owned_dofs(),
                                             mpi_communicator,
                                             locally_relevant_dofs);
  
  system_matrix_poisson.clear(); // store the matrix that will be solved in the newton iterations
  system_matrix_poisson.reinit(locally_owned_dofs, locally_owned_dofs, dsp,  mpi_communicator);

  laplace_matrix.clear(); //store laplace matrix
  laplace_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp,  mpi_communicator);

  mass_matrix.clear();  //store mass matrix
  mass_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp,  mpi_communicator);

  density_matrix.clear(); // store the term: M(n+p)q0/V_TH in the newton problem
  density_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp,  mpi_communicator);
  
  // DYNAMIC SPARSITY PATTERN DRIFT DIFFUSION PROBLEM AND RELATIVE MATRICES
  DynamicSparsityPattern dsp_dd(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp_dd, density_constraints, false); //noi mettavamo true;  prima era constraints e basta

  SparsityTools::distribute_sparsity_pattern(dsp_dd,
                                             dof_handler.locally_owned_dofs(),
                                             mpi_communicator,
                                             locally_relevant_dofs);
  
  hole_drift_diffusion_matrix.clear();
  hole_drift_diffusion_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp_dd,  mpi_communicator);

  electron_drift_diffusion_matrix.clear();
  electron_drift_diffusion_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp_dd,  mpi_communicator);

  hole_matrix.clear();
  hole_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp_dd,  mpi_communicator);

  electron_matrix.clear();
  electron_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp_dd,  mpi_communicator);


  // pcout << " End of setup_system "<< std::endl<<std::endl;

}

//---------------------------------------------------------------------------------------------------------------------------------------------------

// template <int dim>
// void PoissonProblem<dim>:: initialize_current_solution(){

//   //NB: the L2 norm of current solution without constraints is zero, it's correct (without = 0).
//   //NB: abbiamo constatato che setup_system scrive effettivamente in constraints i valori imposti
//   //    ergo in constrints ho effettivamente i valori delle BC legate ai vari dof dei processori VEDERE CARTELLA OUTPUT AFFINE CONST STEP PRECEDENTE.
//   //    Ciononostante noi poi non usiamo constraints per imporre le boundary conditions
//   //NB: finalmente in current solution ci sono dei valori sensati, una volta impostate le BCs ha norma L2 non banale e 
//   //    norma L_INF coerente con i valori delle BCs.  QUESTA FUNCTION FA IL SUO
//   //    unica cosa brutta l'uso di un temp vector
 
//   PETScWrappers::MPI::Vector temp;
// 	temp.reinit(locally_owned_dofs, mpi_communicator);   //non ghosted, serve per imporre i valori delle BCs
// 	temp = current_solution; //current solution here is zero by default constructor
	  
//   std::map<types::global_dof_index, double> boundary_values;
  
//   VectorTools::interpolate_boundary_values(dof_handler,
//                                            1,
//                                            Functions::ConstantFunction<dim>(V_TH*std::log(D/ni)),
//                                            boundary_values);

//   VectorTools::interpolate_boundary_values(dof_handler,
//                                            2,
//                                            Functions::ConstantFunction<dim>(-V_TH*std::log(A/ni)),
//                                            boundary_values);
  
//   for (auto &boundary_value : boundary_values){
//     temp(boundary_value.first) = boundary_value.second;
//   }
  
//   temp.compress(VectorOperation::insert); //giusto insert, add non funziona
//   current_solution = temp;
  
//   pcout << " The L2 norm of current solution is: "<< current_solution.l2_norm()<< std::endl;
//   pcout << " The L_INF norm of current solution is: "<< current_solution.linfty_norm()<< std::endl;
  
//   pcout << " End of initialization_current_solution "<< std::endl<<std::endl;


// }
//-------------------------------------------------------------------------------------------------------------------------------------------------
/*template <int dim>
void PoissonProblem<dim>:: compute_densities(){   
  
  // in this function we compute the densities of electrons and holes starting from current solution that is the potential

  PETScWrappers::MPI::Vector temp1;
  PETScWrappers::MPI::Vector temp2;
  
	temp1.reinit(locally_owned_dofs, mpi_communicator);
  temp2.reinit(locally_owned_dofs, mpi_communicator);
  
	// temp1 = current_solution;
  // temp2 = current_solution;
  
  //double check = 0;

  // Get the local indices
  const IndexSet& local_elements = temp1.locally_owned_elements();
  
  for (const auto& i : local_elements){ 

    temp1[i] = ni*std::exp(current_solution[i]/V_TH); //electrons
    temp2[i] = ni*std::exp(-current_solution[i]/V_TH); //holes
    
    //check = temp1[i]*temp2[i]; deve essere 10^20 --> lo stampa giusto
    //pcout << "check densità: " <<check << std::endl;
  }
  
  // Make sure to have updated ghost values to synchronize across MPI processes
  temp1.update_ghost_values();
  temp2.update_ghost_values();


  temp1.compress(VectorOperation::insert);
  temp2.compress(VectorOperation::insert);
  
  electron_density = temp1;
  hole_density = temp2;
  
  pcout << " The L2 norm of electorn density is: "<< electron_density.l2_norm()<< std::endl;
  pcout << " The L_INF norm of electron density is: "<< electron_density.linfty_norm()<< std::endl;

  pcout << " The L2 norm of hole density is: "<< hole_density.l2_norm()<< std::endl;
  pcout << " The L_INF norm of hole density is: "<< hole_density.linfty_norm()<< std::endl;

  pcout << " End of compute densities "<< std::endl<<std::endl;

}*/
//-----------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void PoissonProblem<dim>::initialization()
{

  PETScWrappers::MPI::Vector temp_elec;
  PETScWrappers::MPI::Vector temp_hole;
  PETScWrappers::MPI::Vector temp_pot;

	temp_elec.reinit(locally_owned_dofs, mpi_communicator); 
  temp_hole.reinit(locally_owned_dofs, mpi_communicator); 
  temp_pot.reinit(locally_owned_dofs, mpi_communicator); 

  MappingQ1<dim> mapping;

  VectorTools::interpolate(mapping, dof_handler, ElectronInitialValues<dim>(), temp_elec);
  VectorTools::interpolate(mapping, dof_handler, HoleInitialValues<dim>(), temp_hole);
  VectorTools::interpolate(mapping, dof_handler, PotentialValues<dim>(), temp_pot);

  constraints.distribute(temp_pot);
  density_constraints.distribute(temp_hole);
  density_constraints.distribute(temp_elec);
	 
  old_electron_density = temp_elec;
  old_hole_density = temp_hole;
  current_solution = temp_pot;

  // pcout << "end of initialization" <<std::endl;
}

//-----------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void PoissonProblem<dim>::assemble_laplace_matrix(bool use_nonzero_constraints)
{
	const QTrapezoid<dim> quadrature_formula;

	laplace_matrix = 0;

	FEValues<dim> fe_values(fe,
							quadrature_formula,
							update_values | update_gradients |
							update_quadrature_points | update_JxW_values);

	const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
	const unsigned int n_q_points    = quadrature_formula.size();

	FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	for (const auto &cell : dof_handler.active_cell_iterators()){
	  if (cell->is_locally_owned()){
		    cell_matrix = 0.;

        fe_values.reinit(cell);

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              cell_matrix(i, j) += fe_values.shape_grad(i, q_point) * fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point);
            }
          }

        const AffineConstraints<double> &constraints_used =
            use_nonzero_constraints ? constraints : zero_constraints;

        cell->get_dof_indices(local_dof_indices);
        constraints_used.distribute_local_to_global(cell_matrix,
                                                    local_dof_indices,
                                                    laplace_matrix);
		  }
  }

laplace_matrix.compress(VectorOperation::add);

pcout << " The L_INF norm of the laplace matrix is "<<laplace_matrix.linfty_norm() <<std::endl;
pcout << " The L_FROB norm of the laplace matrix is "<<laplace_matrix.frobenius_norm() <<std::endl<<std::endl;
// pcout << " End of Assembling Laplace matrix "<< std::endl<<std::endl;
}

//---------------------------------------------------------------------------------------------------------------------------------------------------

template <int dim>
void PoissonProblem<dim>::assemble_mass_matrix(bool use_nonzero_constraints)
{
	const QTrapezoid<dim> quadrature_formula;

	mass_matrix = 0;

	FEValues<dim> fe_values(fe,
							quadrature_formula,
							update_values | update_gradients |
							update_quadrature_points | update_JxW_values);

	const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
	const unsigned int n_q_points    = quadrature_formula.size();

	FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	for (const auto &cell : dof_handler.active_cell_iterators()){
	  if (cell->is_locally_owned())
		{
		cell_matrix = 0.;

		fe_values.reinit(cell);

		for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
			{
			for (unsigned int i = 0; i < dofs_per_cell; ++i)
				{
				for (unsigned int j = 0; j < dofs_per_cell; ++j)
					cell_matrix(i, j) += fe_values.shape_value(i, q_point) * fe_values.shape_value(j, q_point) * fe_values.JxW(q_point);
				}
			}

    const AffineConstraints<double> &constraints_used =
            use_nonzero_constraints ? constraints : zero_constraints;

		cell->get_dof_indices(local_dof_indices);
		constraints_used.distribute_local_to_global(cell_matrix,
												                        local_dof_indices,
												                        mass_matrix );
		}
  }

	mass_matrix.compress(VectorOperation::add);
  pcout << " The L_INF norm of the mass matrix is "<<mass_matrix.linfty_norm() <<std::endl;
  pcout << " The L_FROB norm of the mass matrix is "<<mass_matrix.frobenius_norm() <<std::endl<<std::endl;
  // pcout << " End of Assembling Mass matrix "<< std::endl<<std::endl;

}

//-------------------------------------------------------------------------------------------------------------------------------------------------------

template <int dim>
void PoissonProblem<dim>::assemble_nonlinear_poisson(bool use_nonzero_constraints)
{
  //BUILDING SYSTEM MATRIX

  // initialize both matrices 
  system_matrix_poisson = 0;
  density_matrix = 0;
  
  // temporary vectors that stores old densities
  PETScWrappers::MPI::Vector temp_elec; //temporary non-ghosted vector that stores electorn_density
  PETScWrappers::MPI::Vector temp_hole; //temporary non-ghosted vector that stores hole_density
  
	temp_elec.reinit(locally_owned_dofs, mpi_communicator);
  temp_hole.reinit(locally_owned_dofs, mpi_communicator);

  temp_elec = old_electron_density;
  temp_hole = old_hole_density;

  double new_value = 0;
  
  // generate the term:  (n+p)*MASS_MAT   lumped version stored in density_matrix

  pcout << " The L_INF norm of the mass_matrix is "<<mass_matrix.linfty_norm() <<std::endl;
  pcout << " The L_INF norm of the temp_elec is "<<temp_elec.linfty_norm() <<std::endl;

  for (auto iter = locally_owned_dofs.begin(); iter != locally_owned_dofs.end(); ++iter){  

    new_value = mass_matrix(*iter, *iter) * (temp_elec[*iter] + temp_hole[*iter]);

    density_matrix.set(*iter,*iter,new_value);

  }

  density_matrix.compress(VectorOperation::insert);
  
  pcout << " The L_INF norm of the density matrix is "<<density_matrix.linfty_norm() <<std::endl;
  pcout << " The L_FROB norm of the density matrix is "<<density_matrix.frobenius_norm() <<std::endl<<std::endl;

  system_matrix_poisson.add(eps_r * eps_0, laplace_matrix); //ho checkato che passi giusto.  This term is: SYS_MAT = SYS_MAT +  eps*A

  system_matrix_poisson.add(q0 / V_TH, density_matrix);   // SYS_MAT = SYS_MAT + q0/V_TH * (n+p)*MASS_MAT

  
  pcout << " The L_INF norm of the assembled matrix is "<<system_matrix_poisson.linfty_norm() <<std::endl;
  pcout << " The L_FROB norm of the assembled matrix is "<<system_matrix_poisson.frobenius_norm() <<std::endl<<std::endl;

  
  // BUILDING  POISSON SYSTEM RHS

  poisson_system_rhs = 0;

  PETScWrappers::MPI::Vector temp;
  PETScWrappers::MPI::Vector doping;     //store the term N, that is 1e+22 on the left side and 1e-22 on the right
   
	temp.reinit(locally_owned_dofs, mpi_communicator);
  doping.reinit(locally_owned_dofs, mpi_communicator);
  
  doping = 0;
  temp = 0;

  MappingQ1<dim> mapping;

  VectorTools::interpolate(mapping, dof_handler, DopingValues<dim>(), doping); // We interpolate the previusly created vector with the initial values of Doping provided by DopingValues

  doping += temp_hole;
  doping -= temp_elec;

  // basically: temp = (N -n +p)

  mass_matrix.vmult(temp,doping);  // temp = MASS*(N -n +p)

  poisson_system_rhs.add(q0, temp);     // SYS_RHS = q0*MASS*(N -n +p)

  laplace_matrix.vmult(temp, current_solution); // temp = A*phi

  poisson_system_rhs.add(- eps_r * eps_0, temp);    //SYS_RHS = SYS_RHS - eps*A*phi

  const AffineConstraints<double> &constraints_used =
            use_nonzero_constraints ? constraints : zero_constraints;

  constraints_used.distribute(poisson_system_rhs);

  // pcout<<" End assemble of non linear poisson problem"<<std::endl;

}
//------------------------------------------------------------------------------------------------------------------------------------------------

template <int dim>
void PoissonProblem<dim>::solve_poisson(bool use_nonzero_constraints)
{
  // NB: Assemble funziona e riempie con elementi dell'ordine di 1e-10 sia la matrice che il rhs del system
  // NB: Solve sembrerebbe funzionare, non sono sicuro su quali setting mettere e quale solver usare,
  //     ciononostante otteniamo una current solution con una normal L2 diversa( maggiore) di quella dopo l'inizializzazione (L_INF norm uguale)
  
  SolverControl sc_p(dof_handler.n_dofs(), 1e-10);     
  PETScWrappers::SparseDirectMUMPS solverMUMPS(sc_p);     
  solverMUMPS.solve(system_matrix_poisson, newton_update, poisson_system_rhs);

  const AffineConstraints<double> &constraints_used =
            use_nonzero_constraints ? constraints : zero_constraints;

  constraints_used.distribute(newton_update);

  //CLUMPING
  double result=0;
  
  // for (auto iter = locally_owned_dofs.begin(); iter != locally_owned_dofs.end(); ++iter){ 
  for (const auto& iter : locally_owned_dofs){ 

    if (newton_update[iter] < -V_TH) {
      result = -V_TH;
    } else if (newton_update[iter] > V_TH) {
      result = V_TH;
    } else {
      result = newton_update[iter];
    }

    newton_update[iter] = result;

  }

  newton_update.compress(VectorOperation::insert); 
  
  PETScWrappers::MPI::Vector temp;
  temp.reinit(locally_owned_dofs, mpi_communicator);

  temp = current_solution;
  temp.add(0.8, newton_update); 
  // per adesso noi aggiorniamo cosi: phi_k+1 = phi_k + a * delta_phi. dove a =1, ma è una scelta
  current_solution = temp;

  pcout << "L2 norm of the current solution: " << current_solution.l2_norm() << std::endl;
  pcout << "L_INF norm of the current solution: " << current_solution.linfty_norm() << std::endl;

  // pcout << " End of solve "<< std::endl<<std::endl;
  
}
//--------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void PoissonProblem<dim>::one_cycle_newton_poisson(bool use_nonzero_constraints,
                                                   const unsigned int max_iter_newton, 
                                                   const double toll_newton){

  unsigned int counter = 0; // it keeps track of newton iteration

  pcout << " -- START ONE CYCLE OF NEWTON METHOD --"<< std::endl<<std::endl<<std::endl;

  double increment_norm = std::numeric_limits<double>::max(); // the increment norm is + inf

  pcout << " Initial Newton Increment Norm dphi: " << increment_norm << std::endl<<std::endl;
  
  //bool non_zero_constraints; //in order to cuse_nonzero_constraintshoose the BCs

  while(counter < max_iter_newton && increment_norm > toll_newton){ //METTERE LE MATRICI FUORIII

    pcout << " NEWTON ITERATION NUMBER: "<< counter +1<<std::endl<<std::endl;
    pcout << " Assemble System Poisson Matrix"<< std::endl;

    if(counter == 0){

      assemble_laplace_matrix(use_nonzero_constraints);
      assemble_mass_matrix(use_nonzero_constraints);

      assemble_nonlinear_poisson(use_nonzero_constraints);

    }else{
     
      assemble_laplace_matrix(false);
      assemble_mass_matrix(false);

      assemble_nonlinear_poisson(false);
      
    }
    
    
    
    //solve newton iteration and update the potential
    // pcout << " Solve Newton cycle"<< std::endl;
    if(counter == 0){

      solve_poisson(use_nonzero_constraints); // dentro c'e anche il clamping

    }else{
     
      solve_poisson(false);

    }
    

    //update the densities
    PETScWrappers::MPI::Vector old_temp_elec;
    PETScWrappers::MPI::Vector old_temp_hole;

    old_temp_elec.reinit(locally_owned_dofs, mpi_communicator);
    old_temp_hole.reinit(locally_owned_dofs, mpi_communicator);

    old_temp_elec  = old_electron_density;
    old_temp_hole  = old_hole_density;


    const IndexSet& local_elements = old_temp_elec.locally_owned_elements();
    

    for (const auto& i : local_elements){ 
    
      old_temp_elec(i) *= std::exp(newton_update(i)/V_TH);   //bisogna imporre constraints ?
      old_temp_hole(i) *= std::exp(-newton_update(i)/V_TH);

    }
    
    // Make sure to have updated ghost values to synchronize across MPI processes
    old_temp_elec.update_ghost_values();
    old_temp_hole.update_ghost_values();

    old_temp_elec.compress(VectorOperation::insert);
    old_temp_hole.compress(VectorOperation::insert);
    
    old_electron_density = old_temp_elec;
    old_hole_density = old_temp_hole;

    increment_norm = newton_update.l2_norm();
    pcout << " Update Increment: "<<increment_norm<<std::endl<<std::endl;
    counter ++;

  }

  if(counter == max_iter_newton){
    pcout<< " MAX NUMBER OF NEWTON ITERATIONS REACHED!"<<std::endl;
  }


  // pcout << " end one cycle of newton method "<< std::endl<<std::endl;

}
//-----------------------------------------------------------------------------------------------------------------------------------------------------------

template <int dim>
void PoissonProblem<dim>::assemble_drift_diffusion_matrix()
{

	rhs_electron_density = 0;
	rhs_hole_density = 0;
	// hole_drift_diffusion_matrix = 0;
	// electron_drift_diffusion_matrix = 0;
  hole_matrix = 0;
	electron_matrix = 0;

  const unsigned int vertices_per_cell = 4; // 4 sono anche i dof for cell
  std::vector<types::global_dof_index> local_dof_indices(vertices_per_cell);

  const unsigned int t_size = 3;

  Vector<double> cell_rhs(t_size);
  FullMatrix<double> A(t_size,t_size), B(t_size,t_size), neg_A(t_size,t_size), neg_B(t_size,t_size);

  std::vector<types::global_dof_index> A_local_dof_indices(t_size);
  std::vector<types::global_dof_index> B_local_dof_indices(t_size);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned()){

        A = 0;
        B = 0;
        neg_A = 0;
        neg_B = 0;
        cell_rhs = 0;
        
        cell->get_dof_indices(local_dof_indices);

        // Lexicographic ordering
        const Point<dim> v1 = cell->vertex(2); // top left
        const Point<dim> v2 = cell->vertex(3); // top right
        const Point<dim> v3 = cell->vertex(0); // bottom left
        const Point<dim> v4 = cell->vertex(1); // bottom right

        const double u1 = -current_solution[local_dof_indices[2]]/V_TH;
        const double u2 = -current_solution[local_dof_indices[3]]/V_TH;
        const double u3 = -current_solution[local_dof_indices[0]]/V_TH;
        const double u4 = -current_solution[local_dof_indices[1]]/V_TH;

        const double l_alpha = side_length(v1,v4);
        const double l_beta = side_length(v2,v3);
        
        const double alpha21 = (u1 - u2);
        const double alpha42 = (u2 - u4);
        const double alpha34 = (u4 - u3);
        const double alpha13 = (u3 - u1);
        
        const double neg_alpha21 =  - (u1 - u2);
        const double neg_alpha42 =  - (u2 - u4);
        const double neg_alpha34 = - (u4 - u3);
        const double neg_alpha13 = - (u3 - u1);

        if (l_alpha >= l_beta) { // l_alpha is the longest diagonal: split by beta
              const double alpha23 =  (u3 - u2);
              const double neg_alpha23 = - (u3 - u2);
              //cout << "Alpha 23 is: " << alpha23 << endl;
              
              // Triangle A:
              A= compute_triangle_matrix(v2,v1,v3, alpha21, alpha13, -alpha23, Dp);
              neg_A= compute_triangle_matrix(v2,v1,v3, neg_alpha21, neg_alpha13, -neg_alpha23, Dn);
              
              // Triangle B:
              B = compute_triangle_matrix(v3,v4,v2, alpha34, alpha42, alpha23, Dp);
              neg_B = compute_triangle_matrix(v3,v4,v2, neg_alpha34, neg_alpha42, neg_alpha23, Dn);
              
              // Matrix assemble
              A_local_dof_indices[0] = local_dof_indices[3];
              A_local_dof_indices[1] = local_dof_indices[2];
              A_local_dof_indices[2] = local_dof_indices[0];
              
              B_local_dof_indices[0] = local_dof_indices[0];
              B_local_dof_indices[1] = local_dof_indices[1];
              B_local_dof_indices[2] = local_dof_indices[3];

            } else { // l_beta is the longest diagonal: split by alpha
              const double alpha14 = (u4 - u1);
              const double neg_alpha14 = - (u4 - u1);
              //cout << "Alpha 14 is: " << alpha14 << endl;
              
              // Triangle A:
              A = compute_triangle_matrix(v4,v2,v1, alpha42, alpha21, alpha14, Dp);
              neg_A = compute_triangle_matrix(v4,v2,v1, neg_alpha42, neg_alpha21, neg_alpha14, Dn);
              
              // Triangle B:
              B = compute_triangle_matrix(v1,v3,v4, alpha13, alpha34, -alpha14, Dp);
              neg_B = compute_triangle_matrix(v1,v3,v4, neg_alpha13, neg_alpha34, -neg_alpha14, Dn);
              
              A_local_dof_indices[0] = local_dof_indices[1];
              A_local_dof_indices[1] = local_dof_indices[3];
              A_local_dof_indices[2] = local_dof_indices[2];
              
              B_local_dof_indices[0] = local_dof_indices[2];
              B_local_dof_indices[1] = local_dof_indices[0];
              B_local_dof_indices[2] = local_dof_indices[1];
              
				     }
        
				constraints.distribute_local_to_global(A, cell_rhs,  A_local_dof_indices, hole_matrix, rhs_hole_density);
				constraints.distribute_local_to_global(B, cell_rhs,  B_local_dof_indices, hole_matrix, rhs_hole_density);

				constraints.distribute_local_to_global(neg_A, cell_rhs,  A_local_dof_indices, electron_matrix, rhs_electron_density);
				constraints.distribute_local_to_global(neg_B, cell_rhs,  B_local_dof_indices, electron_matrix, rhs_electron_density);

        }
		  }
    
    
    hole_matrix.compress(VectorOperation::add);
    electron_matrix.compress(VectorOperation::add);
    
    rhs_hole_density.compress(VectorOperation::add);
    rhs_electron_density.compress(VectorOperation::add);

    // pcout << " End of assembling drift diffusion matrix"<< std::endl<<std::endl;

}
//-----------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void PoissonProblem<dim>::apply_drift_diffusion_boundary_conditions() //forse non serve per le nostre inizializzazioni
{
  PETScWrappers::MPI::Vector temp_elec;
  PETScWrappers::MPI::Vector temp_hole;     
   
	temp_elec.reinit(locally_owned_dofs, mpi_communicator);
  temp_hole.reinit(locally_owned_dofs, mpi_communicator);

  temp_elec = electron_density;
  temp_hole = hole_density;

  MappingQ1<dim> mapping;
  
  std::map<types::global_dof_index, double> emitter_boundary_values, collector_boundary_values;

  VectorTools::interpolate_boundary_values(mapping, dof_handler,1, Functions::ConstantFunction<dim>(N1), emitter_boundary_values);
  MatrixTools::apply_boundary_values(emitter_boundary_values, electron_matrix, temp_elec, rhs_electron_density);

  VectorTools::interpolate_boundary_values(mapping, dof_handler,2, Functions::ConstantFunction<dim>(N2), collector_boundary_values);
  MatrixTools::apply_boundary_values(collector_boundary_values, electron_matrix, temp_elec, rhs_electron_density);

  VectorTools::interpolate_boundary_values(mapping, dof_handler,1, Functions::ConstantFunction<dim>(P1), emitter_boundary_values);
  MatrixTools::apply_boundary_values(emitter_boundary_values, hole_matrix, temp_hole, rhs_hole_density);

  VectorTools::interpolate_boundary_values(mapping, dof_handler,2, Functions::ConstantFunction<dim>(P2), collector_boundary_values);
  MatrixTools::apply_boundary_values(collector_boundary_values, hole_matrix, temp_hole, rhs_hole_density);
  
  electron_density = temp_elec;
  hole_density = temp_hole;

  // pcout<<" end of apply boundary conditions for drift diffusion"<<std::endl<<std::endl;

}
//-----------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void PoissonProblem<dim>::solve_drift_diffusion()
{
  PETScWrappers::MPI::Vector temp_elec;
  PETScWrappers::MPI::Vector temp_hole;

  temp_elec.reinit(locally_owned_dofs, mpi_communicator);
  temp_hole.reinit(locally_owned_dofs, mpi_communicator);
  
  SolverControl sc_hole(dof_handler.n_dofs(), 1e-10;
  SolverControl sc_elec(dof_handler.n_dofs(), 1e-10); 
  
  PETScWrappers::SparseDirectMUMPS solverMUMPS_hole(sc_hole); 
  PETScWrappers::SparseDirectMUMPS solverMUMPS_elec(sc_elec); 



  pcout << "L_INF norm of the hole_matrix: " << hole_matrix.linfty_norm() << std::endl;
  pcout << "L_INF norm of the electron_matrix: " << electron_matrix.linfty_norm() << std::endl;


  pcout << "L_INF norm of the rhs_hole_density: " << rhs_hole_density.linfty_norm() << std::endl;
  pcout << "L_INF norm of the rhs_electron_density: " << rhs_electron_density.linfty_norm() << std::endl;

  solverMUMPS_hole.solve(hole_matrix, temp_hole, rhs_hole_density);
  solverMUMPS_elec.solve(electron_matrix, temp_elec, rhs_electron_density);

  constraints.distribute(temp_hole);
  constraints.distribute(temp_elec);

  hole_density = temp_hole;
  electron_density = temp_elec;

  pcout << "L_INF norm of the hole_density: " << hole_density.linfty_norm() << std::endl;
  pcout << "L_INF norm of the electron_density: " << electron_density.linfty_norm() << std::endl;


  // pcout << " End of solve drift diffusion"<< std::endl<<std::endl;
}
//-----------------------------------------------------------------------------------------------------------------------------------------------------

template <int dim>
void PoissonProblem<dim>::output_results(const unsigned int cycle)
{

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(current_solution, "phi");
  data_out.add_data_vector(electron_density, "n");
  data_out.add_data_vector(hole_density, "p");
  

  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record(
    "./", "solution", cycle, mpi_communicator, 2, 1);


}

//---------------------------------------------------------------------------------------------------------------------------------------------------

template <int dim>
void PoissonProblem<dim>::run() 
{
  unsigned int cycle_drift_diffusion = 0;
  unsigned int max_drift_diffusion_iterations = 20;
  double hole_tol = 1.e-10;
  double electron_tol = 1.e-10;

  double hole_err = hole_tol + 1.;
  double electron_err = electron_tol + 1.;

  unsigned int max_newton_iterations = 80;
  double tol_newton = 1e-10;

  pcout << "-- START OF DRIFT DIFFUSION PROBLEM --" <<std::endl<<std::endl;

  pcout<< " Setup system" <<std::endl;
  setup_system();
  PETScWrappers::MPI::Vector temp;
	temp.reinit(locally_owned_dofs, mpi_communicator);

  pcout<< " Inizialization of variables" <<std::endl;
  initialization();
  
  output_results(cycle_drift_diffusion); // in order to see initial conditions
  
  // ho fatto un giro di run fino a questo punto e le IC con le relative BCs sono giuste

  bool use_nonzero_constraints = true;
  
  while ( (hole_err > hole_tol || electron_err > electron_tol) && cycle_drift_diffusion < max_drift_diffusion_iterations)
  {
    ++cycle_drift_diffusion;
    pcout<< " CYCLE NUMBER " <<cycle_drift_diffusion<<" OF DRIFT DIFFUSION"<<std::endl;
    
    pcout<< " NEWTON ITERATION FOR POISSON" <<std::endl;
    one_cycle_newton_poisson(use_nonzero_constraints, max_newton_iterations, tol_newton);

    pcout<< " Assemble drift diffusion matrix" <<std::endl;
    assemble_drift_diffusion_matrix();

    // hole_matrix.copy_from(hole_drift_diffusion_matrix);
    // electron_matrix.copy_from(electron_drift_diffusion_matrix);

    pcout<< " apply_drift_diffusion_boundary_conditions"<<std::endl;
    apply_drift_diffusion_boundary_conditions();

    pcout<< " solve_drift_diffusion"<<std::endl;
    solve_drift_diffusion();

    pcout<< "Update error for convergence"<<std::endl;

    electron_tol = 1.e-10*old_electron_density.linfty_norm();
    hole_tol = 1.e-10*old_hole_density.linfty_norm();

    temp = hole_density;
    temp -= old_hole_density;
    hole_err = temp.linfty_norm();

    temp = electron_density;
    temp -= old_electron_density;
    electron_err = temp.linfty_norm();

    output_results(cycle_drift_diffusion);

    old_hole_density = hole_density;
    old_electron_density = electron_density;

    // pcout<< " ## ERRORS ##:"<<std::endl;
    pcout<<" ELECTRON DENSITY ERROR:"<< electron_err<<std::endl;
    pcout<<" HOLE DENSITY ERROR:"<< hole_err<<std::endl;
    
    use_nonzero_constraints = false;
  }
  
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
//---------------------------------------------------------------------------------------------------------------------------------------------------------------
//COMMENTO: il codice gira ma 1) non rispetta BCs di nessuno
//                            2) errore delle densità enorme
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
      
      poisson_problem_2d.run();
      
    
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
