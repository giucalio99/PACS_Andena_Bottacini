#ifndef BLOCKSCHURPRECONDITIONER_HPP
#define BLOCKSCHURPRECONDITIONER_HPP

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

using namespace dealii;

// @sect3{Block preconditioner}
  //
  // The block Schur preconditioner can be written as the product of three
  // matrices:
  // $
  //   P^{-1} = \begin{pmatrix} \tilde{A}^{-1} & 0\\ 0 & I\end{pmatrix}
  //            \begin{pmatrix} I & -B^T\\ 0 & I\end{pmatrix}
  //            \begin{pmatrix} I & 0\\ 0 & \tilde{S}^{-1}\end{pmatrix}
  // $
  // $\tilde{A}$ is symmetric since the convection term is eliminated from the
  // LHS.
  // $\tilde{S}^{-1}$ is the inverse of the Schur complement of $\tilde{A}$,
  // which consists of a reaction term, a diffusion term, a Grad-Div term
  // and a convection term.
  // In practice, the convection contribution is ignored, namely
  // $\tilde{S}^{-1} = -(\nu + \gamma)M_p^{-1} -
  //                   \frac{1}{\Delta{t}}{[B(diag(M_u))^{-1}B^T]}^{-1}$
  // where $M_p$ is the pressure mass, and
  // ${[B(diag(M_u))^{-1}B^T]}$ is an approximation to the Schur complement of
  // (velocity) mass matrix $BM_u^{-1}B^T$.
  //
  // Same as the tutorials, we define a vmult operation for the block
  // preconditioner
  // instead of write it as a matrix. It can be seen from the above definition,
  // the result of the vmult operation of the block preconditioner can be
  // obtained
  // from the results of the vmult operations of $M_u^{-1}$, $M_p^{-1}$,
  // $\tilde{A}^{-1}$, which can be transformed into solving three symmetric
  // linear
  // systems.
class BlockSchurPreconditioner : public Subscriptor
{
public:
  BlockSchurPreconditioner(
    TimerOutput &timer,      
    double gamma,
    double viscosity,
    double dt, 
    const std::vector<IndexSet> &owned_partitioning,
    const PETScWrappers::MPI::BlockSparseMatrix &system,
    const PETScWrappers::MPI::BlockSparseMatrix &mass,
    PETScWrappers::MPI::BlockSparseMatrix &schur);

  void vmult(PETScWrappers::MPI::BlockVector &dst,
              const PETScWrappers::MPI::BlockVector &src) const;

private:
  TimerOutput &timer;          //This class can be used to generate formatted output from time measurements of different subsections in a program
  const double gamma;          //Parameter of Grad-div stabilization 
  const double viscosity;
  const double dt;

  const SmartPointer<const PETScWrappers::MPI::BlockSparseMatrix>
    system_matrix;
  const SmartPointer<const PETScWrappers::MPI::BlockSparseMatrix> mass_matrix;
  // As discussed, ${[B(diag(M_u))^{-1}B^T]}$ and its inverse
  // need to be computed.
  // We can either explicitly compute it out as a matrix, or define
  // it as a class with a vmult operation.
  // The second approach saves some computation to construct the matrix,
  // but leads to slow convergence in CG solver because it is impossible
  // to apply a preconditioner. We go with the first route.
  const SmartPointer<PETScWrappers::MPI::BlockSparseMatrix> mass_schur;    //Schur complement of the velocity mass
};

#include "BlockSchurPreconditioner_impl.hpp"

#endif