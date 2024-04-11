#ifndef PROBLEM_HPP
#define PROBLEM_HPP

#include "Electrical_Values.hpp" // to use the override template functions (Electrical_Values) and the constants (Electrical_Constants)

// These are all the include that MatMes wrote in the orignal code; may be some redundancies in the inclusions

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

//To parallelize
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/lac/petsc_block_sparse_matrix.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/petsc_solver.h>   //Add to use MUMps direct solver

#include <fstream>
#include <cmath>

// Template class Problem, it needs a dimension as template input, in our case is 2
template <int dim>
class Problem{

  public:

    // CONSTRUCTOR
    Problem(parallel::distributed::Triangulation<dim> &triangulation);

    // PUBLIC METHOD TO RUN THE SOLVER
    void run();

  private:

    // PRIVATE DATA MEMBERS

    // Elements that define the mesh and FEM order
    parallel::distributed::Triangulation<dim> triangulation; //This's a collection of cells that, jointly, cover the domain on which one typically wants to solve a partial differential equation. 
    FE_Q<dim>       fe;               //Implementation of a scalar Lagrange finite element Qp that yields the finite element space of continuous, piecewise polynomials of degree p in each coordinate direction.
    DoFHandler<dim> dof_handler;      //Given a triangulation and a description of a finite element, this class enumerates degrees of freedom on all vertices, edges, faces, and cells of the triangulation. As a result, it also provides a basis for a discrete space, which dof live on which cell
    MappingQ1<dim>  mapping;          //The mapping implemented by this class maps the reference (unit) cell to a general grid cell with straight lines in d dimensions.
    unsigned int step_number;
    
    // AffineConstraints: This class deal with constrains on dof (eg: imposing Dir BCs we are constraining the value of the dof on the boundary)
    AffineConstraints<double> constraints;
    AffineConstraints<double> constraints_poisson;
    AffineConstraints<double> zero_constraints_poisson;
    
    // SparseMatrix: This class implements the functionality to store matrix entry values in the locations denoted by a SparsityPattern.
    // The elements of a SparsityPattern, corresponding to the places where SparseMatrix objects can store nonzero entries, are stored row-by-row
    PETScWrappers::MPI::SparseMatrix laplace_matrix_poisson;
    PETScWrappers::MPI::SparseMatrix mass_matrix_poisson;
    PETScWrappers::MPI::SparseMatrix system_matrix_poisson;
    SparsityPattern      sparsity_pattern_poisson;  

    PETScWrappers::MPI::SparseMatrix ion_system_matrix;
    PETScWrappers::MPI::SparseMatrix mass_matrix;
    PETScWrappers::MPI::SparseMatrix drift_diffusion_matrix;

    PETScWrappers::MPI::SparseMatrix electron_system_matrix;
    PETScWrappers::MPI::SparseMatrix electron_drift_diffusion_matrix;
    SparsityPattern      sparsity_pattern;
    
    // Vector: A class that represents a vector of numerical elements
    PETScWrappers::MPI::Vector poisson_newton_update;
    PETScWrappers::MPI::Vector potential;
    PETScWrappers::MPI::Vector poisson_rhs;

    PETScWrappers::MPI::Vector old_ion_density;
    PETScWrappers::MPI::Vector ion_density;
    PETScWrappers::MPI::Vector ion_rhs;

    PETScWrappers::MPI::Vector old_electron_density;
    PETScWrappers::MPI::Vector electron_density;
    PETScWrappers::MPI::Vector electron_rhs;
    
    
    // Timer: A class that provide a way to measure the CPU time
    Timer timer;

    //To parallelize
    IndexSet local_owned_dofs;
	  IndexSet locally_relevant_dofs;
    MPI_Comm mpi_communicator;

    ConditionalOStream pcout;                     //A class allows you to print an output stream, useful in parallel computations

    // MESH
    void create_mesh();   
    
    // POISSON PROBLEM
    void setup_poisson();
    void assemble_nonlinear_poisson();
    void solve_poisson();
    void newton_iteration_poisson(const double tol, const unsigned int max_iterations);
    
    // DRIFIT DIFFUSION PROBLEM
    void setup_drift_diffusion();
    void assemble_drift_diffusion_matrix();
    void apply_drift_diffusion_boundary_conditions();
    void solve_drift_diffusion();
    
    // OUTPUT
    void output_results(const unsigned int step);
};


// HELPER FUNCTION ( andrebbero modificate e messi punti con reference)
void bernoulli (double x, double &bp, double &bn);
double side_length (const Point<2> a, const Point<2> b);
double triangle_denom(const Point<2> a, const Point<2> b, const Point<2> c);
Tensor<1,2> face_normal(const Point<2> a, const Point<2> b);
FullMatrix<double> compute_triangle_matrix(const Point<2> a, const Point<2> b, const Point<2> c, const double alpha12, const double alpha23, const double alpha31, const double D);

#include "Problem_impl.hpp" //templates implementations


#endif //PROBLEM_HPP