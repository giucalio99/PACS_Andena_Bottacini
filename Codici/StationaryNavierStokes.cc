/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2008 - 2022 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 *
 * Author: Liang Zhao and Timo Heister, Clemson University, 2016
 */

// @sect3{Include files}

// As usual, we start by including some well-known files:
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/tensor.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h> // For Gmesh
#include <deal.II/grid/manifold_lib.h> // To use manifolds

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

// This file includes UMFPACK: the direct solver:
#include <deal.II/lac/sparse_direct.h>

// And the one for ILU preconditioner:
#include <deal.II/lac/sparse_ilu.h>


#include <fstream>
#include <iostream>

// Geometry Data
const double R = 2.5e-4; // [m] emitter edge radius
const double L = 2.*R; // [m] emitter length
const double X = -L/2.; // [m] emitter center

const double g = 0.02; // [m] interelectrode distance
const double collector_length = 0.01; ; // [m]
const double collector_height = 0.002;
const double mesh_height = 0.02;; // [m]

// Physical Data
const double V = 1.; // [m/s] inlet velocity
const double g_y = 9.807; // gravitational acceleration [m/s^2]
const double q0 = 1.602; // 10^-19 [C]
const double rho = 1.225; // kg m^-3
const double E_ON = 3.31e+6; // V/m air electrical breakdown field

namespace Step57
{
  using namespace dealii;
  using namespace std;

  double get_emitter_height(const double &p)
  {
  	if (p <= X-L/2. || p >= X+L/2.)
  		return 0.;

  	double y = 0;
  	double x = 0;

  	const double left_center = X - L/2. + R;
  	const double right_center = X + L/2. - R;

  	if (p <= left_center)
  		x = p - left_center;
  	else if (p >= right_center)
  		x = p - right_center;

  	x /= R;
  	y = R*std::sqrt( 1. - x * x);

  	return y;
  }

  template <int dim>
  class StationaryNavierStokes
  {
  public:
    StationaryNavierStokes();
    void run();

  private:
    void create_mesh();

    void setup_dofs();

    void initialize_system();

    void assemble_system(const bool initial_step);

    void solve(const bool initial_step);

    void output_results(const unsigned int step) const;

    void newton_iteration(const double       tolerance,
                          const unsigned int max_n_line_searches);


    double                               viscosity;
    double                               gamma;
    const unsigned int                   degree;
    std::vector<types::global_dof_index> dofs_per_block;

    MappingQ1<dim> mapping;

    Triangulation<dim> triangulation;
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
  };

  // Collector Manifold - START

   double get_collector_height(const double &p)
   {
   	if (p <= g || p >= g + collector_length)
   		return 0.;

   	const double a = collector_length/2.;
   	const double b = collector_height;

   	return b*std::sqrt(1.-p/a);
   }



     template <int dim>
     class CollectorGeometry : public ChartManifold<dim, dim, dim-1>
     {
     public:
   	virtual Point<dim-1> pull_back(const Point<dim> &space_point) const override;

   	virtual Point<dim> push_forward(const Point<dim-1> &chart_point) const override;

   	virtual std::unique_ptr<Manifold<dim, dim>> clone() const override;

     };

     template <int dim>
     std::unique_ptr<Manifold<dim, dim>> CollectorGeometry<dim>::clone() const
     {
   	return std::make_unique<CollectorGeometry<dim>>();
     }

     template <int dim>
     Point<dim> CollectorGeometry<dim>::push_forward(const Point<dim-1>  &x) const
     {
   	const double y = get_collector_height(x[0]);

   	Point<dim> p;
   	p[0] = x[0]; p[1] = y;

   	if (dim == 3) {
   		p[2] = x[1];
   	}

   	return p;
     }

     template <int dim>
     Point<dim-1>  CollectorGeometry<dim>::pull_back(const Point<dim> &p) const
     {
   	Point<dim-1> x;
   	x[0] = p[0];

   	if (dim == 3) {
   		x[1] = p[2];
   	}

   	return x;
     }
   // Collector Manifold - END

     template <int dim>
      class BoundaryValues : public Function<dim>
      {
      public:
        BoundaryValues()
          : Function<dim>(dim + 1)
        {}
        virtual double value(const Point<dim> & p,
                             const unsigned int component) const override;
      };

      template <int dim>
      double BoundaryValues<dim>::value(const Point<dim> & /*p*/,
                                        const unsigned int component) const
      {
        Assert(component < this->n_components,
               ExcIndexRange(component, 0, this->n_components));

       /* const double H = mesh_height;
        const double h_vmax = 2. * H / 3.;
        const double vel = V * p(1) / h_vmax * p(1) / h_vmax * (H - p(1)) / (H - h_vmax);*/

        if (component == 0) {
        	return V;
        }

        if (component == dim)
        		return 0.;

        return 0.;
      }


  template <class PreconditionerMp>
  class BlockSchurPreconditioner : public Subscriptor
  {
  public:
    BlockSchurPreconditioner(double                           gamma,
                             double                           viscosity,
                             const BlockSparseMatrix<double> &S,
                             const SparseMatrix<double> &     P,
                             const PreconditionerMp &         Mppreconditioner);

    void vmult(BlockVector<double> &dst, const BlockVector<double> &src) const;

  private:
    const double                     gamma;
    const double                     viscosity;
    const BlockSparseMatrix<double> &stokes_matrix;
    const SparseMatrix<double> &     pressure_mass_matrix;
    const PreconditionerMp &         mp_preconditioner;
    SparseDirectUMFPACK              A_inverse;
  };

  // We can notice that the initialization of the inverse of the matrix at the
  // top left corner is completed in the constructor. If so, every application
  // of the preconditioner then no longer requires the computation of the
  // matrix factors.

  template <class PreconditionerMp>
  BlockSchurPreconditioner<PreconditionerMp>::BlockSchurPreconditioner(
    double                           gamma,
    double                           viscosity,
    const BlockSparseMatrix<double> &S,
    const SparseMatrix<double> &     P,
    const PreconditionerMp &         Mppreconditioner)
    : gamma(gamma)
    , viscosity(viscosity)
    , stokes_matrix(S)
    , pressure_mass_matrix(P)
    , mp_preconditioner(Mppreconditioner)
  {
    A_inverse.initialize(stokes_matrix.block(0, 0));
  }

  template <class PreconditionerMp>
  void BlockSchurPreconditioner<PreconditionerMp>::vmult(
    BlockVector<double> &      dst,
    const BlockVector<double> &src) const
  {
    Vector<double> utmp(src.block(0));

    {
        const double tol = 1e-6 * src.block(1).l2_norm(); // Increased from 1.e-6
        const unsigned int Nmax = 1e+4;

      SolverControl solver_control(Nmax, tol); // Increased from 1.e-6
      SolverCG<Vector<double>> cg(solver_control);

      dst.block(1) = 0.0;
      cg.solve(pressure_mass_matrix,
               dst.block(1),
               src.block(1),
               mp_preconditioner);
      dst.block(1) *= -(viscosity + gamma);

      if (solver_control.last_step() >= Nmax -1)
    	  cerr << "Warning! CG has reached the maximum number of iterations " << solver_control.last_step() << " intead of reching the tolerance " << tol << endl;
    }

    {
      stokes_matrix.block(0, 1).vmult(utmp, dst.block(1));
      utmp *= -1.0;
      utmp += src.block(0);
    }

    A_inverse.vmult(dst.block(0), utmp);
  }

  template <int dim>
    void StationaryNavierStokes<dim>::create_mesh()
    {
	  const std::string filename = "./Meshes/BOXED_ELLIPSE.msh";
	  	cout << "Reading from " << filename << endl;
	  	std::ifstream input_file(filename);
	  	GridIn<2>       grid_in;
	  	grid_in.attach_triangulation(triangulation);
	  	grid_in.read_msh(input_file);

	  	const types::manifold_id emitter = 1;
	  	const Point<dim> center(X,0.);
	  	SphericalManifold<2> emitter_manifold(center);

	  	const types::manifold_id collector = 2;
	  	CollectorGeometry<2> collector_manifold;

	  	triangulation.set_all_manifold_ids_on_boundary(1, emitter);
	  	triangulation.set_manifold(emitter, emitter_manifold);
	  	triangulation.set_all_manifold_ids_on_boundary(2, collector);
	  	triangulation.set_manifold(collector, collector_manifold);
	  	cout  << "Active cells: " << triangulation.n_active_cells() << endl;
    }

  template <int dim>
  StationaryNavierStokes<dim>::StationaryNavierStokes()
    : viscosity(2.e-5) // 17.89e-6 [m^2/s^2]
    , gamma(V) // should be the same order of magnitude as V
    , degree(1)
    , triangulation(Triangulation<dim>::maximum_smoothing)
    , NS_fe(FE_Q<dim>(degree + 1), dim, FE_Q<dim>(degree), 1)
    , NS_dof_handler(triangulation)
  	, mapping()
  {}

  template <int dim>
  void StationaryNavierStokes<dim>::setup_dofs()
  {
    NS_dof_handler.distribute_dofs(NS_fe);

    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;
    DoFRenumbering::component_wise(NS_dof_handler, block_component);

    dofs_per_block = DoFTools::count_dofs_per_fe_block(NS_dof_handler, block_component);
    unsigned int dof_u = dofs_per_block[0];
    unsigned int dof_p = dofs_per_block[1];

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar vertical_velocity(1);
    const FEValuesExtractors::Vector vertical_velocity_and_pressure(1);
    {
      nonzero_NS_constraints.clear();

      DoFTools::make_hanging_node_constraints(NS_dof_handler, nonzero_NS_constraints);
      VectorTools::interpolate_boundary_values(NS_dof_handler,
                                                0,
                                                Functions::ZeroFunction<dim>(dim+1),
                                                nonzero_NS_constraints,
                                                NS_fe.component_mask(vertical_velocity));
      VectorTools::interpolate_boundary_values(NS_dof_handler,
                                                1,
                                                Functions::ZeroFunction<dim>(dim+1),
                                                nonzero_NS_constraints,
                                                NS_fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(NS_dof_handler,
                                                2,
                                                Functions::ZeroFunction<dim>(dim+1),
                                                nonzero_NS_constraints,
                                                NS_fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(NS_dof_handler,
                                                10,
												 BoundaryValues<dim>(),
                                                nonzero_NS_constraints,
                                                NS_fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(NS_dof_handler,
                                               11, // Outlet
											   Functions::ZeroFunction<dim>(dim+1),
											   nonzero_NS_constraints,
											   NS_fe.component_mask(vertical_velocity_and_pressure));
    }
    nonzero_NS_constraints.close();

    {
      zero_NS_constraints.clear();

      DoFTools::make_hanging_node_constraints(NS_dof_handler, zero_NS_constraints);
      VectorTools::interpolate_boundary_values(NS_dof_handler,
                                                0,
                                                Functions::ZeroFunction<dim>(dim+1),
                                                zero_NS_constraints,
                                                NS_fe.component_mask(vertical_velocity));
      VectorTools::interpolate_boundary_values(NS_dof_handler,
                                                1,
                                                Functions::ZeroFunction<dim>(dim+1),
                                                zero_NS_constraints,
                                                NS_fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(NS_dof_handler,
                                                2,
                                                Functions::ZeroFunction<dim>(dim+1),
                                                zero_NS_constraints,
                                                NS_fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(NS_dof_handler,
                                                10,
                                                Functions::ZeroFunction<dim>(dim+1),
                                                zero_NS_constraints,
                                                NS_fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(NS_dof_handler,
                                               11, // Outlet
											   Functions::ZeroFunction<dim>(dim+1),
											   zero_NS_constraints,
											   NS_fe.component_mask(vertical_velocity_and_pressure));
    }
    zero_NS_constraints.close();

    std::cout << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Number of degrees of freedom: " << NS_dof_handler.n_dofs()
              << " (" << dof_u << " + " << dof_p << ')' << std::endl;
  }

  template <int dim>
  void StationaryNavierStokes<dim>::initialize_system()
  {
    {
      BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
      DoFTools::make_sparsity_pattern(NS_dof_handler, dsp, nonzero_NS_constraints);
      NS_sparsity_pattern.copy_from(dsp);
    }

    NS_system_matrix.reinit(NS_sparsity_pattern);

    NS_solution.reinit(dofs_per_block);
    NS_newton_update.reinit(dofs_per_block);
    NS_system_rhs.reinit(dofs_per_block);
  }

  template <int dim>
  void StationaryNavierStokes<dim>::assemble_system(const bool initial_step)
  {
    NS_system_matrix = 0;
    NS_system_rhs = 0;

    QGauss<dim> quadrature_formula(degree + 2);

    FEValues<dim> fe_values(NS_fe,
                            quadrature_formula,
                            update_values | update_quadrature_points |
                              update_JxW_values | update_gradients);

    const unsigned int dofs_per_cell = NS_fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<Tensor<1, dim>> present_velocity_values(n_q_points);
    std::vector<Tensor<2, dim>> present_velocity_gradients(n_q_points);
    std::vector<double>         present_pressure_values(n_q_points);

    std::vector<double>         div_phi_u(dofs_per_cell);
    std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
    std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
    std::vector<double>         phi_p(dofs_per_cell);

    Tensor<1,dim> f;

    for (const auto &cell : NS_dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);

        local_matrix = 0;
        local_rhs    = 0;

        fe_values[velocities].get_function_values(NS_solution, //evaluation_point,
                                                  present_velocity_values);

        fe_values[velocities].get_function_gradients(NS_solution, present_velocity_gradients);// evaluation_point, present_velocity_gradients);

        fe_values[pressure].get_function_values(NS_solution, //evaluation_point,
                                                present_pressure_values);

        // Only used in the complete code, can be tested here
        const double E_x = 0.;
        const double E_y = 0.;
        f[0] = q0 * E_x / rho;
        f[1] = q0 * E_y / rho; //  - g_y; // [m/s^2]

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
                div_phi_u[k]  = fe_values[velocities].divergence(k, q);
                grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                phi_u[k]      = fe_values[velocities].value(k, q);
                phi_p[k]      = fe_values[pressure].value(k, q);
              }

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {

                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                        local_matrix(i, j) +=
                          (viscosity *
                             scalar_product(grad_phi_u[j], grad_phi_u[i]) +
                           present_velocity_gradients[q] * phi_u[j] * phi_u[i] +
                           grad_phi_u[j] * present_velocity_values[q] *
                             phi_u[i] -
                           div_phi_u[i] * phi_p[j] - phi_p[i] * div_phi_u[j] +
                           gamma * div_phi_u[j] * div_phi_u[i]
						+ phi_p[i] * phi_p[j]) * // To assemble the pressure mass matrix
                          fe_values.JxW(q);
                      }

                double present_velocity_divergence = trace(present_velocity_gradients[q]);

                local_rhs(i) +=
                  (-viscosity * scalar_product(present_velocity_gradients[q], grad_phi_u[i])
                   - present_velocity_gradients[q] * present_velocity_values[q] * phi_u[i]
				   + present_pressure_values[q] * div_phi_u[i]
				   + present_velocity_divergence * phi_p[i]
				   + scalar_product(phi_u[i], f) // ADDED
				   - gamma * present_velocity_divergence * div_phi_u[i]) * fe_values.JxW(q);
              }
          }

        cell->get_dof_indices(local_dof_indices);

        const AffineConstraints<double> &constraints_used =
          initial_step ? nonzero_NS_constraints : zero_NS_constraints;


            constraints_used.distribute_local_to_global(local_matrix,
                                                        local_rhs,
                                                        local_dof_indices,
                                                        NS_system_matrix,
                                                        NS_system_rhs);

      }


        // Finally we move pressure mass matrix into a separate matrix:
        pressure_mass_matrix.reinit(NS_sparsity_pattern.block(1, 1));
        pressure_mass_matrix.copy_from(NS_system_matrix.block(1, 1));

        NS_system_matrix.block(1, 1) = 0;

  }


  template <int dim>
  void StationaryNavierStokes<dim>::solve(const bool initial_step)
  {
    const AffineConstraints<double> &constraints_used =
      initial_step ? nonzero_NS_constraints : zero_NS_constraints;

    std::cout << "RHS norms are: " << NS_system_rhs.block(0).linfty_norm() << " and " <<  NS_system_rhs.block(1).linfty_norm() << std::endl;

    const double tol = 1e-4 * NS_system_rhs.l2_norm(); // Increased from 1.e-4

    SolverControl solver_control(NS_system_matrix.m(),
                                 tol,
                                 true);

    SolverFGMRES<BlockVector<double>> gmres(solver_control);
    SparseILU<double>                 pmass_preconditioner;
    pmass_preconditioner.initialize(pressure_mass_matrix,
                                    SparseILU<double>::AdditionalData());

    const BlockSchurPreconditioner<SparseILU<double>> preconditioner(
      gamma,
      viscosity,
      NS_system_matrix,
      pressure_mass_matrix,
      pmass_preconditioner);

    gmres.solve(NS_system_matrix, NS_newton_update, NS_system_rhs, preconditioner);
    std::cout << "FGMRES steps: " << solver_control.last_step() << std::endl;

    constraints_used.distribute(NS_newton_update);
  }

  template <int dim>
  void StationaryNavierStokes<dim>::newton_iteration(
    const double       tolerance,
    const unsigned int max_n_line_searches)
  {
        unsigned int line_search_n = 0;
        double       current_res   = 1.0 + tolerance;

        while ((current_res > tolerance) && line_search_n < max_n_line_searches)
          {
			assemble_system(false);
			solve(false);

			const double alpha = 1.;
			NS_solution.add(alpha, NS_newton_update);
			nonzero_NS_constraints.distribute(NS_solution);
			current_res = NS_newton_update.block(0).linfty_norm();
			std::cout << "  alpha: " << std::setw(10) << alpha
					  << std::setw(0) << "  residual: " << current_res
					  << " for " << line_search_n << " line searches"
					  << std::endl;

			++line_search_n;
			output_results(line_search_n);
		  }
  }

  template <int dim>
  void StationaryNavierStokes<dim>::output_results(
    const unsigned int output_index) const
  {
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);
    DataOut<dim> data_out;
    data_out.attach_dof_handler(NS_dof_handler);
    data_out.add_data_vector(NS_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out.build_patches();

    std::ofstream output("Re"+ std::to_string(V * collector_length / viscosity) + "-solution-" +
                         Utilities::int_to_string(output_index, 4) + ".vtk");
    data_out.write_vtk(output);
  }

  // @sect4{StationaryNavierStokes::run}
  //
  // This is the last step of this program. In this part, we generate the grid
  // and run the other functions respectively. The max refinement can be set by
  // the argument.
  template <int dim>
  void StationaryNavierStokes<dim>::run()
  {
    create_mesh();

    const double target_Re = V * collector_length  / viscosity;

    const double newton_tol = 1.e-5; // Increased from 1.e-12
    const unsigned int max_it = 100;

    double current_res;

    setup_dofs();
    initialize_system();

    const double step_size = 1000.0;

    if (target_Re <= 1000.) {
        viscosity = V * collector_length / target_Re;
        std::cout << "Searching for solution with Re = " << target_Re  << std::endl;
        assemble_system(true);
        solve(true);
        NS_solution = NS_newton_update;
        nonzero_NS_constraints.distribute(NS_solution);
        current_res = NS_newton_update.block(0).linfty_norm();
        std::cout << "The residual of initial guess is " << current_res << std::endl;
        std::cout << "Starting newton iteration ... " << std::endl;
        newton_iteration(newton_tol,max_it);
    } else {

		for (double Re = std::min(1000.0,target_Re); Re < target_Re;
			 Re        = std::min(Re + step_size, target_Re))
		  {
			viscosity = V * collector_length / Re;
			std::cout << "Searching for guess with Re = " << Re  << std::endl;
			assemble_system(true);
			solve(true);
			NS_solution = NS_newton_update;
			nonzero_NS_constraints.distribute(NS_solution);
			current_res = NS_newton_update.block(0).linfty_norm();
			std::cout << "The residual of initial guess is " << current_res << std::endl;
			std::cout << "Starting newton iteration ... " << std::endl;
			newton_iteration(newton_tol,max_it);
		  }
    }
  }
} // namespace Step57

int main()
{
  try
    {
      using namespace Step57;

      StationaryNavierStokes<2> flow;
      flow.run();
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
