#ifndef BOUNDARYVALUES_HPP
#define BOUNDARYVALUES_HPP

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

// This file includes UMFPACK: the direct solver:
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

namespace dealii;

// @sect3{Boundary values}
// Dirichlet boundary conditions for the velocity inlet and walls.
template <int dim>
class BoundaryValues : public Function<dim>
  {
  public:
    BoundaryValues() : Function<dim>(dim + 1) {}
    virtual double value(const Point<dim> &p,
                         const unsigned int component) const override;

    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &values) const override;
  };

template <int dim>
double BoundaryValues<dim>::value(const Point<dim> &p,
                                    const unsigned int component) const
  {
    Assert(component < this->n_components,
           ExcIndexRange(component, 0, this->n_components));
    double left_boundary = (dim == 2 ? 0.3 : 0.0);
    if (component == 0 && std::abs(p[0] - left_boundary) < 1e-10)
      {
        // For a parabolic velocity profile, $U_\mathrm{avg} = 2/3
        // U_\mathrm{max}$
        // in 2D, and $U_\mathrm{avg} = 4/9 U_\mathrm{max}$ in 3D.
        // If $\nu = 0.001$, $D = 0.1$, then $Re = 100 U_\mathrm{avg}$.
        double Uavg = 1.0;
        double Umax = (dim == 2 ? 3 * Uavg / 2 : 9 * Uavg / 4);
        double value = 4 * Umax * p[1] * (0.41 - p[1]) / (0.41 * 0.41);
        if (dim == 3)
          {
            value *= 4 * p[2] * (0.41 - p[2]) / (0.41 * 0.41);
          }
        return value;
      }
    return 0;
  }

template <int dim>
void BoundaryValues<dim>::vector_value(const Point<dim> &p,
                                         Vector<double> &values) const
  {
    for (unsigned int c = 0; c < this->n_components; ++c)
      values(c) = BoundaryValues<dim>::value(p, c);
  }

#endif