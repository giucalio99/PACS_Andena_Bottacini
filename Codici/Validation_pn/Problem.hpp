#ifndef PROBLEM.HPP
#define PROBLEM.HPP

#include <deal.II/lac/vector.h>  // to use Vector template from deal ii
#include "Electrical_Values.hpp" // to use the override template functions and the constants

// Template class Problem 
template <int dim>
class Problem
{
public:

  //CONSTRUCTOR
  Problem();

  // PUBLIC METHOD TO RUN THE SOLVER
  void run();

private:

  void create_mesh();

  void setup_poisson();
  void assemble_nonlinear_poisson();
  void solve_poisson();
  void newton_iteration_poisson(const double tol, const unsigned int max_iterations);

  void setup_drift_diffusion();
  void assemble_drift_diffusion_matrix();
  void apply_drift_diffusion_boundary_conditions();
  void solve_drift_diffusion();

  void output_results(const unsigned int step);

  Triangulation<dim> triangulation;

  FE_Q<dim>       fe;
  DoFHandler<dim> dof_handler;
  MappingQ1<dim>  mapping;

  AffineConstraints<double> constraints;
  AffineConstraints<double> constraints_poisson;
  AffineConstraints<double> zero_constraints_poisson;

  SparseMatrix<double> laplace_matrix_poisson;
  SparseMatrix<double> mass_matrix_poisson;
  SparseMatrix<double> system_matrix_poisson;
  SparsityPattern      sparsity_pattern_poisson;

  SparseMatrix<double> ion_system_matrix;
  SparseMatrix<double> mass_matrix;
  SparseMatrix<double> drift_diffusion_matrix;

  SparseMatrix<double> electron_system_matrix;
  SparseMatrix<double> electron_drift_diffusion_matrix;
  SparsityPattern      sparsity_pattern;

  Vector<double> poisson_newton_update;
  Vector<double> potential;
  Vector<double> poisson_rhs;

  Vector<double> old_ion_density;
  Vector<double> ion_density;
  Vector<double> ion_rhs;

  Vector<double> old_electron_density;
  Vector<double> electron_density;
  Vector<double> electron_rhs;

  unsigned int step_number;

  Timer timer;
};


#endif //PROBLEM.HPP