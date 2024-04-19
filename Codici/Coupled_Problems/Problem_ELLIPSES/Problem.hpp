#ifndef "PROBLEM_HPP"
#define "PROBLEM_HPP"

#include "Constants.hpp"
#include "NS_Preconditioner_and_Boundaries_values.hpp"
#include "Geometry.hpp"

using namespace dealii;
using namespace std;

template <int dim>
class Problem
{
public:

  Problem();

  void run();
  
private:

  Triangulation<dim> triangulation;

  Vector<double> Field_X;
  Vector<double> Field_Y;

  FE_Q<dim>       fe;
  DoFHandler<dim> dof_handler;
  MappingQ<dim>  mapping;

  AffineConstraints<double> ion_constraints;
  AffineConstraints<double> constraints_poisson;
  AffineConstraints<double> zero_constraints_poisson;

  SparseMatrix<double> laplace_matrix_poisson;
  SparseMatrix<double> mass_matrix_poisson;
  SparseMatrix<double> system_matrix_poisson;
  SparsityPattern      sparsity_pattern_poisson;

  SparseMatrix<double> ion_system_matrix;
  SparseMatrix<double> ion_mass_matrix;
  SparseMatrix<double> drift_diffusion_matrix;
  SparsityPattern      sparsity_pattern;

  Vector<double> poisson_newton_update;
  Vector<double> potential;
  Vector<double> poisson_rhs;

  Vector<double> old_ion_density;
  Vector<double> ion_density;
  Vector<double> ion_rhs;
  Vector<double> eta;

  Vector<double> Vel_X;
  Vector<double> Vel_Y;
  Vector<double> pressure;

  Vector<double> current_values;

  double                               viscosity;
  double                               gamma;
  const unsigned int                   degree;
  std::vector<types::global_dof_index> dofs_per_block;

  MappingQ1<dim> NS_mapping;

  FESystem<dim>      NS_fe;
  DoFHandler<dim>    NS_dof_handler;

  AffineConstraints<double> zero_NS_constraints;
  AffineConstraints<double> nonzero_NS_constraints;

  BlockSparsityPattern      NS_sparsity_pattern;
  BlockSparseMatrix<double> NS_system_matrix;
  SparseMatrix<double>      pressure_mass_matrix;

  BlockVector<double> NS_solution;
  BlockVector<double> NS_newton_update;
  BlockVector<double> NS_system_rhs;

  unsigned int step_number = 0;
  double timestep = 0;

  Timer timer;

  void create_mesh();

  void setup_poisson();
  void assemble_nonlinear_poisson();
  void solve_poisson();
  void solve_homogeneous_poisson();
  void solve_nonlinear_poisson(const double tol, const unsigned int max_iterations);

  void setup_drift_diffusion(const bool reinitialize_densities);
  void assemble_drift_diffusion_matrix();
  void apply_drift_diffusion_boundary_conditions(Vector<double> &solution);  // commnetata in impl
  void solve_drift_diffusion();
  void perform_drift_diffusion_fixed_point_iteration_step();

  void setup_navier_stokes();
  void assemble_navier_stokes(const bool nonzero_constraints);
  void solve_nonlinear_navier_stokes_step(const bool nonzero_constraints);
  void navier_stokes_newton_iteration( const double tolerance,const unsigned int max_n_line_searches);
  void solve_navier_stokes();
  void estimate_thrust(); // not used   commenatta in impl

  void evaluate_emitter_current(); // optional    commentata in impl
  void evaluate_electric_field();
  void refine_mesh();
  void output_results(const unsigned int step);

};

// HELPER FUNCTIONS ( andrebbero modificate e messi punti con reference)
void bernoulli (double x, double &bp, double &bn);
double side_length (const Point<2> a, const Point<2> b);
double triangle_denom(const Point<2> a, const Point<2> b, const Point<2> c);
Tensor<1,2> face_normal(const Point<2> a, const Point<2> b);
FullMatrix<double> compute_triangle_matrix(const Point<2> a, const Point<2> b, const Point<2> c, const double alpha12, const double alpha23, const double alpha31);

#include "Problem_impl.hpp" // for the implementations of the templates methods

#endif //"PROBLEM_HPP"