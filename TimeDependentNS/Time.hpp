#ifndef TIME_HPP
#define TIME_HPP

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

using namespace dealii;

// @sect3{Time stepping}
// This class is pretty much self-explanatory.
//I create an object of this class that helps me keeping track of the time during the run.
class Time
{
  public:
    Time(const double time_end,
         const double delta_t,
         const double output_interval)
      : timestep(0),
        time_current(0.0),
        time_end(time_end),
        delta_t(delta_t),
        output_interval(output_interval)
    {
    }

    double current() const { return time_current; }
    double end() const { return time_end; }
    double get_delta_t() const { return delta_t; }
    unsigned int get_timestep() const { return timestep; }
    bool time_to_output() const {    unsigned int delta = static_cast<unsigned int>(output_interval / delta_t);            //Number of steps after that an ouput must be generated
                                     return (timestep >= delta && timestep % delta == 0); }
    void increment() {    time_current += delta_t;
                                    ++timestep;}

  private:
    unsigned int timestep;
    double time_current;
    const double time_end;
    const double delta_t;
    const double output_interval;
};

#endif