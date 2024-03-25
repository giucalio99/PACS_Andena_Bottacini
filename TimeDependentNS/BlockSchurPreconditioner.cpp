#include "BlockSchurPreconditioner.hpp"

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


// @sect4{BlockSchurPreconditioner::BlockSchurPreconditioner}
  //
  // Input parameters and system matrix, mass matrix as well as the mass schur
  // matrix are needed in the preconditioner. In addition, we pass the
  // partitioning information into this class because we need to create some
  // temporary block vectors inside.
  BlockSchurPreconditioner::BlockSchurPreconditioner(
    TimerOutput &timer,
    double gamma,
    double viscosity,
    double dt,
    const std::vector<IndexSet> &owned_partitioning,        //Partitioning informations
    const PETScWrappers::MPI::BlockSparseMatrix &system,
    const PETScWrappers::MPI::BlockSparseMatrix &mass,
    PETScWrappers::MPI::BlockSparseMatrix &schur)
    : timer(timer),
      gamma(gamma),
      viscosity(viscosity),
      dt(dt),
      system_matrix(&system),
      mass_matrix(&mass),
      mass_schur(&schur)
  {
    TimerOutput::Scope timer_section(timer, "CG for Sm");                     //This class allows to manage "scoped" sections
    // The schur complemete of mass matrix is actually being computed here.
    PETScWrappers::MPI::BlockVector tmp1, tmp2;
    tmp1.reinit(owned_partitioning, mass_matrix->get_mpi_communicator());      //Create a BlockVector with owned_partitioning.size() blocks, each initialized with the given IndexSet.
    tmp2.reinit(owned_partitioning, mass_matrix->get_mpi_communicator());
    tmp1 = 1;
    tmp2 = 0;
    // Jacobi preconditioner of matrix A is by definition ${diag(A)}^{-1}$,
    // this is exactly what we want to compute.
    PETScWrappers::PreconditionJacobi jacobi(mass_matrix->block(0, 0));     //Take the matrix which is used to form the preconditioner
    jacobi.vmult(tmp2.block(0), tmp1.block(0));                             //Apply the preconditioner to tmp1.block(0) putting the result in tmp2.block(0)
    system_matrix->block(1, 0).mmult(
      mass_schur->block(1, 1), system_matrix->block(0, 1), tmp2.block(0));  //Computation of Schur complement for the mass
  }

  // @sect4{BlockSchurPreconditioner::vmult}
  //
  // The vmult operation strictly follows the definition of
  // BlockSchurPreconditioner
  // introduced above. Conceptually it computes $u = P^{-1}v$. So applied the preconditioner tu u
  void BlockSchurPreconditioner::vmult(
    PETScWrappers::MPI::BlockVector &dst,
    const PETScWrappers::MPI::BlockVector &src) const
  {
    // Temporary vectors
    PETScWrappers::MPI::Vector utmp(src.block(0));
    PETScWrappers::MPI::Vector tmp(src.block(1));
    tmp = 0;
    // This block computes $u_1 = \tilde{S}^{-1} v_1$,
    // where CG solvers are used for $M_p^{-1}$ and $S_m^{-1}$.
    {
      TimerOutput::Scope timer_section(timer, "CG for Mp");
      SolverControl mp_control(src.block(1).size(),
                               1e-6 * src.block(1).l2_norm());
      PETScWrappers::SolverCG cg_mp(mp_control,
                                    mass_schur->get_mpi_communicator());
      // $-(\nu + \gamma)M_p^{-1}v_1$
      PETScWrappers::PreconditionBlockJacobi Mp_preconditioner;
      Mp_preconditioner.initialize(mass_matrix->block(1, 1));
      cg_mp.solve(
        mass_matrix->block(1, 1), tmp, src.block(1), Mp_preconditioner);       //Uses same matrix for which solve and as preconditioner
      tmp *= -(viscosity + gamma);
    }
    // $-\frac{1}{dt}S_m^{-1}v_1$
    {
      TimerOutput::Scope timer_section(timer, "CG for Sm");
      SolverControl sm_control(src.block(1).size(),              //Control class to determine convergence of iterative solvers
                               1e-6 * src.block(1).l2_norm());
      PETScWrappers::SolverCG cg_sm(sm_control,
                                    mass_schur->get_mpi_communicator());
      // PreconditionBlockJacobi works find on Sm if we do not refine the mesh.
      // Because after refine_mesh is called, zero entries will be created on
      // the diagonal (not sure why), which prevents PreconditionBlockJacobi
      // from being used.
      PETScWrappers::PreconditionNone Sm_preconditioner;          //A class that implements a non-preconditioned method
      Sm_preconditioner.initialize(mass_schur->block(1, 1));
      cg_sm.solve(
        mass_schur->block(1, 1), dst.block(1), src.block(1), Sm_preconditioner);
      dst.block(1) *= -1 / dt;
    }
    // Adding up these two, we get $\tilde{S}^{-1}v_1$.
    dst.block(1) += tmp;                                 //in dst.block(1) there's the S applied to src.block(1)
    // Compute $v_0 - B^T\tilde{S}^{-1}v_1$ based on $u_1$.
    system_matrix->block(0, 1).vmult(utmp, dst.block(1));
    utmp *= -1.0;
    utmp += src.block(0);
    // Finally, compute the product of $\tilde{A}^{-1}$ and utmp
    // using another CG solver.
    {
      TimerOutput::Scope timer_section(timer, "CG for A");
      SolverControl a_control(src.block(0).size(),
                              1e-6 * src.block(0).l2_norm());
      PETScWrappers::SolverCG cg_a(a_control,
                                   mass_schur->get_mpi_communicator());
      // We do not use any preconditioner for this block, which is of course
      // slow,
      // only because the performance of the only two preconditioners available
      // PreconditionBlockJacobi and PreconditionBoomerAMG are even worse than
      // none.
      PETScWrappers::PreconditionNone A_preconditioner;
      A_preconditioner.initialize(system_matrix->block(0, 0));
      cg_a.solve(
        system_matrix->block(0, 0), dst.block(0), utmp, A_preconditioner);
    }
  }