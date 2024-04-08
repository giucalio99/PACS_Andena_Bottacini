#ifndef PROBLEM.HPP
#define PROBLEM.HPP

#include <deal.II/lac/vector.h>  // to use Vector template from deal ii
#include "Electrical_Values.hpp" // to use the override template functions (Electrical_Values) and the constants (Electrical_Constants)

// TEMPORANEAMNTE SPOSTATI QUI DAL MAIN PER FARE TEST RUN

#include <deal.II/base/quadrature_lib.h>//This header includes: header "config" that contains all the MACROS, header quadrature and header "points" !
//and many more. NB CONTROLLARE INCLUSIONI HEADER IN MODO TALE DA RAGGIUNGERE TUTTE LE CLASSI
#include <deal.II/base/timer.h> // for the timer

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h> // For UMFPACK
#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>

//#include <deal.II/lac/matrix_out.h> // For matrix output

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h> // For Laplace Matrix

#include <fstream>
#include <cmath>

// Template class Problem, it needs a dimension as template input, in our case is 2
template <int dim>
class Problem{

  public:

    // CONSTRUCTOR
    Problem();

    // PUBLIC METHOD TO RUN THE SOLVER
    void run();

  private:

    // PRIVATE DATA MEMBERS
    Triangulation<dim> triangulation; //This's a collection of cells that, jointly, cover the domain on which one typically wants to solve a partial differential equation. 
    FE_Q<dim>       fe;               //Implementation of a scalar Lagrange finite element Qp that yields the finite element space of continuous, piecewise polynomials of degree p in each coordinate direction.
    DoFHandler<dim> dof_handler;      //Given a triangulation and a description of a finite element, this class enumerates degrees of freedom on all vertices, edges, faces, and cells of the triangulation. As a result, it also provides a basis for a discrete space
    MappingQ1<dim>  mapping;          //The mapping implemented by this class maps the reference (unit) cell to a general grid cell with straight lines in d dimensions.
    unsigned int step_number;

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

    Timer timer;

    // PRIVATE METHODS
    void create_mesh();   //method that create the mesh of the problem, it can take the mesh as input file or generated exploiting deal ii functionalities

    void setup_poisson();
    void assemble_nonlinear_poisson();
    void solve_poisson();
    void newton_iteration_poisson(const double tol, const unsigned int max_iterations);

    void setup_drift_diffusion();
    void assemble_drift_diffusion_matrix();
    void apply_drift_diffusion_boundary_conditions();
    void solve_drift_diffusion();

    void output_results(const unsigned int step);
};


// HELPER FUNCTION
void bernoulli (double x, double &bp, double &bn);
double side_length (const Point<2> a, const Point<2> b);
double triangle_denom(const Point<2> a, const Point<2> b, const Point<2> c);
Tensor<1,2> face_normal(const Point<2> a, const Point<2> b);
FullMatrix<double> compute_triangle_matrix(const Point<2> a, const Point<2> b, const Point<2> c, const double alpha12, const double alpha23, const double alpha31, const double D);


#endif //PROBLEM.HPP