/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2021 by the deal.II authors
 *
 * This file is based on step-6 of the examples section of the deal.II library.
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
 * Author: Matteo Menessini, Politecnico di Milano, 2023
 *
 */

// Time-stepping from step-26

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h> // for the timer

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h> // For neighbor renumbering

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_interface_values.h> // For gradient evaluator

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h> // To use manifolds
#include <deal.II/grid/grid_in.h> // For GMSH
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h> // For UMFPACK
#include <deal.II/lac/solver_gmres.h> // For GMRES
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_ilu.h> // ILU preconditioning

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h> // For Laplace Matrix
#include <deal.II/numerics/fe_field_function.h> // For boundary values
#include <deal.II/numerics/solution_transfer.h> // For the solution transfer
#include <deal.II/numerics/error_estimator.h> // Kelly error estimator

#include <fstream>
#include <cmath>

using namespace dealii;
using namespace std;

// Physical Constants
const double eps_0 = 8.854 * 1.e-12; //[F/m]= [C^2 s^2 / kg / m^3]
const double eps_r = 1.0006;
const double q0 = 1.602 * 1.e-19; // [C]
const double kB = 1.381 * 1.e-23 ; //[J/K]

const bool stratosphere = false; // if false, use atmospheric 0 km conditions

// Input Parameters
const double E_ON = 3.31e+6; // onset field threshold [V/m]
const double E_ref = 7.e+7; // [V/m] maximum field value
const double N_ref = 1.e+13; // [m^-3] maximum density value
const double N_0 = stratosphere ? 2.2e-3 : 0.5e-3; // [m^-3] ambient ion density
const double p_amb = stratosphere ? 5474 : 101325;
const double T = stratosphere ? 217. : 303.; // [K] fluid temperature
const double p_over_rho = 0.;// Boundary condition at fluid outlet
const double delta = p_amb/101325*298/T;
const double rho = stratosphere ? 0.089 : 1.225; // kg m^-3
const double Mm = 29.e-3; // kg m^-3,average air molar mass
const double Avo = 6.022e+23; // Avogadro's number

const double q_over_eps_0 = q0 / eps_0; // [m^3 kg C^-1 s^-2]
const double mu0 = 1.83e-4; // [m^2/s/V] from Moseley
const double mu = mu0 * delta; // scaled mobility from Moseley
const double V_E = kB * T / q0; // [V] ion temperature
const double D = mu * V_E;
//const double n_air = rho / Mm * Avo; // m^-3

// Geometry Data

// emitter
const double Ve = 2.e+4; // [V] emitter voltage
const double Re = 2.5e-5; // [m] emitter radius
const double X = -Re; // [m] emitter center

const double g = 0.02; // [m] interelectrode distance

// collector
const double collector_length = 0.1; // [m]

// Peek's law (empyrical)
const double eps = 1.; // wire surface roughness correction coefficient
const double Ep = E_ON*delta*eps*(1+0.308/std::sqrt(Re*1.e+2*delta));
const double Ri = Ep/E_ON*Re; // [m] ionization radius
const double Vi = Ve - Ep*Re*log(Ep/E_ON); // [V] voltage on ionization region boundary


// Collector Manifold - START
double get_collector_height(const double &X)
{
	const double x = (X-g)/collector_length;
	double y = 0;

	if ( abs(x-1.) > 1e-12 && abs(x) > 1e-12 ) {
		double a0 = 0.2969;
		double a1 = -0.126;
		double a2 = -0.3516;
		double a3 = 0.2843;
		double a4 = -0.1036; // or -0.1015 for an open trailing edge
		double t = 0.5; // Last 2 digits of the NACA divided by 20

		y = t*( a0 * sqrt(x) + a1 * x + a2 * pow(x,2.0) + a3 * pow(x,3.0) + a4 * pow(x,4.0) );
	}

	return y * collector_length;
}

    template <int dim,int sign>
    class CollectorGeometry : public ChartManifold<dim, dim, dim-1>
    {
    public:
  	virtual Point<dim-1> pull_back(const Point<dim> &space_point) const override;

  	virtual Point<dim> push_forward(const Point<dim-1> &chart_point) const override;

  	virtual std::unique_ptr<Manifold<dim, dim>> clone() const override;

    };

    template <int dim,int sign>
    std::unique_ptr<Manifold<dim, dim>> CollectorGeometry<dim,sign>::clone() const
    {
  	return std::make_unique<CollectorGeometry<dim,sign>>();
    }

    template <int dim,int sign>
    Point<dim> CollectorGeometry<dim,sign>::push_forward(const Point<dim-1>  &x) const
    {
  	const double y = sign*get_collector_height( x[0] );

  	Point<dim> p;
  	p[0] = x[0]; p[1] = y;

  	if (dim == 3) {
  		p[2] = x[1];
  	}

  	return p;
    }

    template <int dim, int sign>
    Point<dim-1>  CollectorGeometry<dim, sign>::pull_back(const Point<dim> &p) const
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
class Problem
{
public:
  Problem();

  void run();
  
private:
  void create_mesh();

  void setup_poisson();
  void assemble_nonlinear_poisson();
  void solve_poisson();
  void solve_homogeneous_poisson();
  void solve_nonlinear_poisson(const double tol, const unsigned int max_iterations);

  void setup_drift_diffusion(const bool reinitialize_densities);
  void assemble_drift_diffusion_matrix();
  void apply_drift_diffusion_boundary_conditions(Vector<double> &solution);
  void solve_drift_diffusion();
  void perform_drift_diffusion_fixed_point_iteration_step();

  void setup_navier_stokes();
  void assemble_navier_stokes(const bool nonzero_constraints);
  void solve_nonlinear_navier_stokes_step(const bool nonzero_constraints);
  void navier_stokes_newton_iteration( const double tolerance,const unsigned int max_n_line_searches);
  void solve_navier_stokes();
  void estimate_thrust();

  void evaluate_electric_field();
  void refine_mesh();
  void output_results(const unsigned int step);

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
};

Tensor<1,2> get_emitter_normal(const Point<2> a) {

	Tensor<1,2> normal;

	normal[0] = a[0] - X;
	normal[1] = a[1];

	const double norm = std::sqrt(normal[0]*normal[0]+normal[1]*normal[1]);

	return normal/norm;
}


// Functions for local triangle matrix assembly - START

// Bernoulli function
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


FullMatrix<double> compute_triangle_matrix(const Point<2> a, const Point<2> b, const Point<2> c, const double alpha12, const double alpha23, const double alpha31)
{
	const unsigned int size = 3;
	FullMatrix<double> tria_matrix(size,size);

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

	const double D = mu * V_E;

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
// Functions for local triangle matrix assmbly - END

// NS preconditioner - START
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
        //cout << "Tol for CG is " << tol << endl;
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
  // NS preconditioner - END

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

 if (component == 0) {
	return 1.;
 }

 if (component == dim)
		return p_over_rho;

 return 0.;
}

template <int dim>
Problem<dim>::Problem()
  : fe(1) // linear  elements
  , dof_handler(triangulation)
  , timestep(1.e-5)
  , mapping(1)

  ,	viscosity(stratosphere ? 1.6e-4 : 1.8e-5) //  [m^2/s^2]
  , gamma(1.) // should be the same order of magnitude as the expected velocity
  , degree(1)
  , NS_fe(FE_Q<dim>(degree+1), dim, FE_Q<dim>(degree), 1)
  , NS_dof_handler(triangulation)
  , NS_mapping()
{}


template <int dim>
void Problem<dim>::create_mesh()
{
	const std::string filename =  "../../gmsh/REAL_EMITTER.msh";
	cout << "Reading from " << filename << endl;
	std::ifstream input_file(filename);
	GridIn<2>       grid_in;
	grid_in.attach_triangulation(triangulation);
	grid_in.read_msh(input_file);

	const types::manifold_id emitter = 1;
  	const Point<dim> center(X,0.);
  	SphericalManifold<2> emitter_manifold(center);

  	const int up = 1;
  	const int down = -1;
	const types::manifold_id collector_up = 2;
	CollectorGeometry<2,up> naca_up;
	const types::manifold_id collector_down = 3;
	CollectorGeometry<2,down> naca_down;

	triangulation.set_all_manifold_ids_on_boundary(1, emitter);
	triangulation.set_manifold(emitter, emitter_manifold);
	triangulation.set_all_manifold_ids_on_boundary(2, collector_up);
	triangulation.set_manifold(collector_up, naca_up);
	triangulation.set_all_manifold_ids_on_boundary(3, collector_down);
	triangulation.set_manifold(collector_down, naca_down);

	for (auto &cell : triangulation.active_cell_iterators()) {
		if (cell->at_boundary()) {
			for (const auto &face : cell->face_iterators()) {
				if (face->at_boundary()&& face->boundary_id()==2) {
					for (unsigned int i = 0; i < 2; ++i) {
						face->vertex(i)[1] = get_collector_height(face->vertex(i)[0]);
						//cout << face->vertex(i)[0] << " mapped to " << face->vertex(i)[1] << endl;
					}
		         } else if (face->at_boundary()&& face->boundary_id()==3) {
					for (unsigned int i = 0; i < 2; ++i) {
						face->vertex(i)[1] = -get_collector_height(face->vertex(i)[0]);
						//cout << face->vertex(i)[0] << " mapped to " << face->vertex(i)[1] << endl;
					}
				}

			}

		}
	}

	const unsigned int pre_refinement_steps = 2;

	for (unsigned int i = 0; i < pre_refinement_steps; ++i) {

	  const double threshold = collector_length/20./(i+1);
	  Vector<float> criteria(triangulation.n_active_cells());
	  unsigned int ctr = 0;

		for (auto &cell : triangulation.active_cell_iterators()) {

		  Point<2> c = cell->center();
		  double d = threshold + 1.;
		  if (c[0] <= g + collector_length*0.3 + threshold && c[0] >= g - 3.*threshold)
			  d = std::fabs(c[1]) - get_collector_height(c[0]);

		  if (d < threshold)
			  criteria[ctr++] = 1;
		  else
			  criteria[ctr++] = 0;
		}
	  GridRefinement::refine(triangulation, criteria, 0.5);
	  triangulation.execute_coarsening_and_refinement();
	}

	cout  << "Active cells: " << triangulation.n_active_cells() << endl;
}


template <int dim>
void Problem<dim>::setup_poisson()
{
	dof_handler.distribute_dofs(fe);

    Field_X.reinit(dof_handler.n_dofs());
    Field_Y.reinit(dof_handler.n_dofs());

	potential.reinit(dof_handler.n_dofs());
	poisson_newton_update.reinit(dof_handler.n_dofs());

	constraints_poisson.clear();
	DoFTools::make_hanging_node_constraints(dof_handler, constraints_poisson);
	VectorTools::interpolate_boundary_values(dof_handler, 1, Functions::ConstantFunction<dim>(Vi), constraints_poisson);
	VectorTools::interpolate_boundary_values(dof_handler, 2, Functions::ZeroFunction<dim>(), constraints_poisson);
	VectorTools::interpolate_boundary_values(dof_handler, 3, Functions::ZeroFunction<dim>(), constraints_poisson);
	constraints_poisson.close();

	// Used for the update term in Newton's method
	zero_constraints_poisson.clear();
	DoFTools::make_hanging_node_constraints(dof_handler, zero_constraints_poisson);
	VectorTools::interpolate_boundary_values(dof_handler, 1, Functions::ZeroFunction<dim>(), zero_constraints_poisson);
	VectorTools::interpolate_boundary_values(dof_handler, 2, Functions::ZeroFunction<dim>(), zero_constraints_poisson);
	VectorTools::interpolate_boundary_values(dof_handler, 3, Functions::ZeroFunction<dim>(), zero_constraints_poisson);
	zero_constraints_poisson.close();

	DynamicSparsityPattern dsp(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints_poisson, true);

	sparsity_pattern_poisson.copy_from(dsp);

	system_matrix_poisson.reinit(sparsity_pattern_poisson);

	laplace_matrix_poisson.reinit(sparsity_pattern_poisson);
	MatrixCreator::create_laplace_matrix(mapping, dof_handler, QTrapezoid<dim>(), laplace_matrix_poisson);
	mass_matrix_poisson.reinit(sparsity_pattern_poisson);
	MatrixCreator::create_mass_matrix(mapping, dof_handler, QTrapezoid<dim>(), mass_matrix_poisson);

	poisson_rhs.reinit(dof_handler.n_dofs());
}


template <int dim>
void Problem<dim>::assemble_nonlinear_poisson()
{
  // Assemble mattrix
  system_matrix_poisson = 0;

  SparseMatrix<double> ion_mass_matrix(sparsity_pattern_poisson);
  ion_mass_matrix = 0;

  for (unsigned int i = 0; i < eta.size(); ++i)
	  ion_mass_matrix(i,i) = mass_matrix_poisson(i,i) * eta(i);

  system_matrix_poisson.add(q0 / V_E, ion_mass_matrix);
  system_matrix_poisson.add(eps_r * eps_0, laplace_matrix_poisson);

  // Assmble RHS
  poisson_rhs = 0;

  Vector<double> tmp(dof_handler.n_dofs());
  mass_matrix_poisson.vmult(tmp,eta);
  poisson_rhs.add(q0, tmp);
  laplace_matrix_poisson.vmult(tmp,potential);
  poisson_rhs.add(- eps_r * eps_0, tmp);

  zero_constraints_poisson.condense(system_matrix_poisson, poisson_rhs);
}


template <int dim>
void Problem<dim>::solve_poisson()
{
  SparseDirectUMFPACK A_direct;
  A_direct.initialize(system_matrix_poisson);
  A_direct.vmult(poisson_newton_update, poisson_rhs);

  zero_constraints_poisson.distribute(poisson_newton_update);
}

template <int dim>
void Problem<dim>::solve_nonlinear_poisson(const double tolerance, const unsigned int max_n_line_searches)
  {
	unsigned int line_search_n = 1;
	double current_res =  tolerance + 1;

	while (current_res > tolerance && line_search_n <= max_n_line_searches)
	  {
			assemble_nonlinear_poisson();
			solve_poisson();

			// Update Clamping
			const double alpha = 1.;

			current_res = poisson_newton_update.linfty_norm();
			for (unsigned int i = 0; i < poisson_newton_update.size(); i++) {
				poisson_newton_update(i) = std::max(std::min(poisson_newton_update(i),V_E),-V_E);

				eta(i) *= std::exp(-alpha*poisson_newton_update(i)/V_E);
			}
			ion_constraints.distribute(eta);

			potential.add(alpha, poisson_newton_update);
			constraints_poisson.distribute(potential);

			++line_search_n;
	  }

	if (line_search_n >= max_n_line_searches)
		cout << "WARNING! NLP reached " << max_n_line_searches << " iterations achieving a residual " << current_res << endl;
  }

template <int dim>
void Problem<dim>::solve_homogeneous_poisson()
  {
	VectorTools::interpolate(mapping,dof_handler, Functions::ZeroFunction<dim>(), poisson_rhs);
	system_matrix_poisson.copy_from(laplace_matrix_poisson);
	constraints_poisson.condense(system_matrix_poisson,poisson_rhs);
	solve_poisson();
	potential = poisson_newton_update;
	constraints_poisson.distribute(potential);
  }

template <int dim>
void Problem<dim>::setup_drift_diffusion(const bool reinitialize_densities)
{
	if (reinitialize_densities) {
		eta.reinit(dof_handler.n_dofs());
		ion_density.reinit(dof_handler.n_dofs());
		old_ion_density.reinit(dof_handler.n_dofs());
	}

	// Corona inception condition
   Functions::FEFieldFunction<dim> solution_as_function_object(dof_handler, potential, mapping);
	auto boundary_evaluator = [&] (const Point<dim> &p)
		{
			Tensor<1,dim> grad_U = solution_as_function_object.gradient(p);

			const double EXP = std::exp((grad_U.norm()-E_ON)/E_ref);

			return N_ref * EXP;
		};

	ion_constraints.clear();
	DoFTools::make_hanging_node_constraints(dof_handler, ion_constraints);
	VectorTools::interpolate_boundary_values(dof_handler,1, ScalarFunctionFromFunctionObject<2>(boundary_evaluator), ion_constraints);
	VectorTools::interpolate_boundary_values(dof_handler,10, Functions::ConstantFunction<dim>(N_0), ion_constraints);
	ion_constraints.close();

	DynamicSparsityPattern dsp(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler, dsp, ion_constraints, false);
	sparsity_pattern.copy_from(dsp);

	ion_mass_matrix.reinit(sparsity_pattern);
	MatrixCreator::create_mass_matrix(mapping, dof_handler, QTrapezoid<dim>(),ion_mass_matrix, (const Function<dim> *const)nullptr, ion_constraints);

	ion_rhs.reinit(dof_handler.n_dofs());

	ion_system_matrix.reinit(sparsity_pattern);
	drift_diffusion_matrix.reinit(sparsity_pattern);
}


template <int dim>
void Problem<dim>::assemble_drift_diffusion_matrix()
{
	drift_diffusion_matrix = 0;

  const unsigned int vertices_per_cell = 4;
  FullMatrix<double> Robin(vertices_per_cell,vertices_per_cell);
  
  Vector<double> cell_rhs(vertices_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(vertices_per_cell);
  FullMatrix<double> cell_matrix(vertices_per_cell, vertices_per_cell);

  const unsigned int t_size = 3;
  Vector<double> A_cell_rhs(t_size), B_cell_rhs(t_size);
  FullMatrix<double> A(t_size,t_size), B(t_size,t_size);
  std::vector<types::global_dof_index> A_local_dof_indices(t_size);
  std::vector<types::global_dof_index> B_local_dof_indices(t_size);
  
  evaluate_electric_field();

  const double Vh = std::sqrt(8.*numbers::PI*kB * T / Mm * Avo / 2. / numbers::PI); // Hopf velocity
  //const double Vh = std::sqrt(kB * T / Mm * Avo / 2. / numbers::PI); // Thermal velocity


  QTrapezoid<dim-1>	face_quadrature;
  const unsigned int n_q_points = face_quadrature.size();
  FEFaceValues<dim> face_values(fe, face_quadrature, update_values | update_quadrature_points /*| update_normal_vectors*/ | update_JxW_values);


  for (const auto &cell : dof_handler.active_cell_iterators())
    {
	    A = 0;
	    B = 0;
		cell_matrix = 0;
		cell_rhs = 0;
		Robin = 0;
		A_cell_rhs = 0;
		B_cell_rhs = 0;
		cell->get_dof_indices(local_dof_indices);

		// Robin conditions at outlet and (optional) at collector
		if (cell->at_boundary()) {
			for (const auto &face : cell->face_iterators()) {
				if (face->at_boundary() && face->boundary_id() == 11) {

					face_values.reinit(cell,face);

					for (unsigned int i = 0; i < vertices_per_cell; ++i) {

						const double vel_f = Vel_X(local_dof_indices[i]);

						for (unsigned int q = 0; q < n_q_points; ++q) {
							for (unsigned int j = 0; j < vertices_per_cell; ++j) {
								Robin(i,j) += face_values.JxW(q) * face_values.shape_value(i,q) * face_values.shape_value(j,q) * vel_f;
							}
						}
					}

				} else if (face->at_boundary() && (face->boundary_id() == 2 || face->boundary_id() == 3)) {
					face_values.reinit(cell,face);

					for (unsigned int i = 0; i < vertices_per_cell; ++i) {
						Tensor<1,dim> Evec;
						Evec[0] = Field_X(local_dof_indices[i]);
						Evec[1] = Field_Y(local_dof_indices[i]);

						for (unsigned int q = 0; q < n_q_points; ++q) {

							//const Tensor<1,dim> n = face_values.normal_vector(q); // Normal  pointing into the electrode
							//const double En = Evec * n;

							// Alternatively, as the field should be normal to the electrode (equipotential surface):
							const double En = std::sqrt(Evec[0]*Evec[0] + Evec[1]*Evec[1]);

							// CHECK
							if (std::isnan(En) || En <0.) cout << "WARNING! Scalar product is " << En << " in " << face->center() << endl;

							for (unsigned int j = 0; j < vertices_per_cell; ++j) {
									Robin(i,j) += face_values.JxW(q) * face_values.shape_value(i,q) * face_values.shape_value(j,q) * (Vh + mu * En);
							}
						}
					}
				}
			}
		}
		// End RObin conditions

		// Lexicographic ordering
		const Point<dim> v1 = cell->vertex(2); // top left
		const Point<dim> v2 = cell->vertex(3); // top right
		const Point<dim> v3 = cell->vertex(0); // bottom left
		const Point<dim> v4 = cell->vertex(1); // bottom right

		const double u1 = -potential[local_dof_indices[2]]/V_E;
		const double u2 = -potential[local_dof_indices[3]]/V_E;
		const double u3 = -potential[local_dof_indices[0]]/V_E;
		const double u4 = -potential[local_dof_indices[1]]/V_E;

		const double l_12 = side_length(v1,v2);
		const double l_31 = side_length(v1,v3);
		const double l_24 = side_length(v4,v2);
		const double l_43 = side_length(v3,v4);

		const double l_alpha = std::sqrt(l_12*l_12 + l_24*l_24 - 2*((v1 - v2) * (v4 - v2)) );
		const double l_beta = std::sqrt(l_43*l_43 + l_24*l_24 - 2*((v2 - v4) * (v3 - v4)) );

		Tensor<1,dim> u_f_1, u_f_2, u_f_3, u_f_4;
		u_f_1[0] = Vel_X(local_dof_indices[2]);
		u_f_1[1] = Vel_Y(local_dof_indices[2]);
		u_f_2[0] = Vel_X(local_dof_indices[3]);
		u_f_2[1] = Vel_Y(local_dof_indices[3]);
		u_f_3[0] = Vel_X(local_dof_indices[0]);
		u_f_3[1] = Vel_Y(local_dof_indices[0]);
		u_f_4[0] = Vel_X(local_dof_indices[1]);
		u_f_4[1] = Vel_Y(local_dof_indices[1]);

		const Tensor<1,dim> dir_21 = (v1 - v2)/l_12;
		const Tensor<1,dim> dir_42 = (v2 - v4)/l_24;
		const Tensor<1,dim> dir_34 = (v4 - v3)/l_43;
		const Tensor<1,dim> dir_13 = (v3 - v1)/l_31;

		const double alpha21 = (u_f_2 * dir_21)/D*l_12 + (u1 - u2);
		const double alpha42 = (u_f_4 * dir_42)/D*l_24 + (u2 - u4);
		const double alpha34 = (u_f_3 * dir_34)/D*l_43 + (u4 - u3);
		const double alpha13 = (u_f_1 * dir_13)/D*l_31 + (u3 - u1);

		if (l_alpha >= l_beta) { // l_alpha is the longest diagonal: split by beta
					const double l_23 = side_length(v2,v3);
					const Tensor<1,dim> dir_23 = (v3 - v2)/l_beta;

					const double alpha23 = (u_f_2 * dir_23)/D*l_23 + (u3 - u2);

					// Triangle A:
					A= compute_triangle_matrix(v2,v1,v3, alpha21, alpha13, -alpha23);

					// Triangle B:
					B = compute_triangle_matrix(v3,v4,v2, alpha34, alpha42, alpha23);

					// Matrix assemble
					A_local_dof_indices[0] = local_dof_indices[3];
					A_local_dof_indices[1] = local_dof_indices[2];
					A_local_dof_indices[2] = local_dof_indices[0];

					B_local_dof_indices[0] = local_dof_indices[0];
					B_local_dof_indices[1] = local_dof_indices[1];
					B_local_dof_indices[2] = local_dof_indices[3];


				} else { // l_beta is the longest diagonal: split by alpha
					const double l_14 = side_length(v1,v4);
					const Tensor<1,dim> dir_14 = (v4 - v1)/l_alpha;

					const double alpha14 = (u_f_1 * dir_14)/D*l_14 + (u4 - u1);

					// Triangle A:
					A = compute_triangle_matrix(v4,v2,v1, alpha42, alpha21, alpha14);

					// Triangle B:
					B = compute_triangle_matrix(v1,v3,v4, alpha13, alpha34, -alpha14);

					A_local_dof_indices[0] = local_dof_indices[1];
					A_local_dof_indices[1] = local_dof_indices[3];
					A_local_dof_indices[2] = local_dof_indices[2];

					B_local_dof_indices[0] = local_dof_indices[2];
					B_local_dof_indices[1] = local_dof_indices[0];
					B_local_dof_indices[2] = local_dof_indices[1];
				}

				// As the ion system matrix is M + delta t DD, the contributions are multiplied by the timestep
				for (unsigned int i = 0; i < t_size; ++i) {
					for (unsigned int j = 0; j < t_size; ++j) {
						A(i,j) = A(i,j)*timestep;
						B(i,j) = B(i,j)*timestep;
					}
				}

				for (unsigned int i = 0; i < vertices_per_cell; ++i) {
					for (unsigned int j = 0; j < vertices_per_cell; ++j) {
						Robin(i,j) = Robin(i,j)*timestep;
					}
				}

		        ion_constraints.distribute_local_to_global(A, A_cell_rhs,  A_local_dof_indices, ion_system_matrix, ion_rhs);
				ion_constraints.distribute_local_to_global(B, B_cell_rhs,  B_local_dof_indices, ion_system_matrix, ion_rhs);
				ion_constraints.distribute_local_to_global(Robin,  cell_rhs, local_dof_indices, ion_system_matrix, ion_rhs);
		    }
}

template <int dim>
void Problem<dim>::solve_drift_diffusion()
{
  SparseDirectUMFPACK A_direct;
  A_direct.initialize(ion_system_matrix);
  A_direct.vmult(ion_density, ion_rhs);

  ion_constraints.distribute(ion_density);
}


template <int dim>
 void Problem<dim>::perform_drift_diffusion_fixed_point_iteration_step()
{
    Vector<double> tmp(old_ion_density.size());

	ion_rhs = 0;
	ion_system_matrix = 0;

	ion_mass_matrix.vmult(tmp, old_ion_density);
	ion_rhs += tmp;

	// Integration in time with BE:
	ion_system_matrix.copy_from(ion_mass_matrix);
    assemble_drift_diffusion_matrix();

    // Uncomment to add a non-zero forcing term to DD equations ...
	/*
    Vector<double> forcing_terms(old_ion_density.size());
    VectorTools::create_right_hand_side(dof_handler, QTrapezoid<dim>(), Functions::ZeroFunction<dim>(), tmp); // ... by changing the ZeroFunction to an appropriate one
	forcing_terms = tmp;
	forcing_terms *= timestep;
	ion_rhs += forcing_terms;
	*/

    solve_drift_diffusion();
}


template <int dim>
void Problem<dim>::output_results(const unsigned int step)
{
	const double logN = std::log10(N_ref);
	const double logE = std::log10(E_ref);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(ion_density, "Ion_Density");
    data_out.add_data_vector(potential, "Potential");
    data_out.add_data_vector(pressure, "Pressure");
    data_out.add_data_vector(Vel_X, "Velocity_X");
    data_out.add_data_vector(Vel_Y, "Velocity_Y");
    data_out.add_data_vector(Field_X, "Field_X");
    data_out.add_data_vector(Field_Y, "Field_Y");
    data_out.build_patches();

    std::string filename;
	if ( stratosphere == true) {
		filename = "R_Stratosphere_N" + Utilities::int_to_string(logN, 1) + "_E" + Utilities::int_to_string(logE, 1) +  "_solution-" +  Utilities::int_to_string(step, 3) + ".vtk";
	} else
		filename = "R_Atmosphere_N" + Utilities::int_to_string(logN, 1) + "_E" + Utilities::int_to_string(logE, 1) +  "_solution-" +  Utilities::int_to_string(step, 3) + ".vtk";

    DataOutBase::VtkFlags vtk_flags;
    vtk_flags.compression_level = DataOutBase::VtkFlags::ZlibCompressionLevel::best_speed;
    data_out.set_flags(vtk_flags);
    std::ofstream output(filename);
    data_out.write_vtk(output);


    // NS output:
    /*std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);
    DataOut<dim> data_out_NS;
    data_out_NS.attach_dof_handler(NS_dof_handler);
    data_out_NS.add_data_vector(NS_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out_NS.build_patches();

    std::string NSfile;
	if ( stratosphere == true) {
    			NSfile = "NS_R_Stratosphere_N" + Utilities::int_to_string(logN, 1) + "_E" + Utilities::int_to_string(logE, 1) +  "_solution-" +  Utilities::int_to_string(step, 3) + ".vtk";
    		} else {
    			NSfile = "NS_R_Atmosphere_N" + Utilities::int_to_string(logN, 1) + "_E" + Utilities::int_to_string(logE, 1) +  "_solution-" +  Utilities::int_to_string(step, 3) + ".vtk";
        }

    std::ofstream NSoutput(NSfile);
    data_out_NS.write_vtk(NSoutput);*/
}

template <int dim>
void Problem<dim>::evaluate_electric_field()
{
	Field_X.reinit(dof_handler.n_dofs());
	Field_Y.reinit(dof_handler.n_dofs());

    const unsigned int 		dofs_per_cell = fe.n_dofs_per_cell();
    std::vector<double>		global_dof_hits(dof_handler.n_dofs());

    Vector<double>		el_field_X(dof_handler.n_dofs());
    Vector<double>		el_field_Y(dof_handler.n_dofs());

	QTrapezoid<dim-1>			iv_quadrature;
	FEInterfaceValues<dim> 		fe_iv(fe, iv_quadrature, update_gradients);

	const unsigned int 						n_q_points = iv_quadrature.size();
    std::vector<Tensor<1,dim>> 				iv_gradients(n_q_points);

	std::vector<types::global_dof_index> 	local_dof_indices(dofs_per_cell);

	for (auto &cell : dof_handler.active_cell_iterators())
	  {
		for (const auto face_index : GeometryInfo<dim>::face_indices())
	      {

	        fe_iv.reinit(cell, face_index);
	        local_dof_indices = fe_iv.get_interface_dof_indices();

	        fe_iv.get_average_of_function_gradients(potential, iv_gradients);

	        for (const auto q : fe_iv.quadrature_point_indices()) {
	          for (const auto i : fe_iv.dof_indices()) {

				global_dof_hits[local_dof_indices[i]] += 1.;

				for (unsigned int d = 0; d < dim; ++d) {
					if (d == 0)
						el_field_X(local_dof_indices[i]) += - iv_gradients[q][d]; //-grad_phi_i[d] / n_q_points;
					else if (d == 1)
						el_field_Y(local_dof_indices[i]) +=  - iv_gradients[q][d]; // * (J_inverse[d] * shape_gradient) * dx;
					else
						Assert(false, ExcNotImplemented());
				}

	          }
	        }
	      }
	  }

   // Take the average of all dof values
   for (unsigned int k = 0; k < dof_handler.n_dofs(); k++) {
	   el_field_X(k) /= std::max(1., global_dof_hits[k]);
	   el_field_Y(k) /= std::max(1., global_dof_hits[k]);
   }

   Field_X = el_field_X;
   Field_Y = el_field_Y;
}

template <int dim>
  void Problem<dim>::setup_navier_stokes()
  {
    NS_dof_handler.distribute_dofs(NS_fe);

	Vel_X.reinit(dof_handler.n_dofs());
	Vel_Y.reinit(dof_handler.n_dofs());
	pressure.reinit(dof_handler.n_dofs());

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
                                                3,
                                                Functions::ZeroFunction<dim>(dim+1),
                                                nonzero_NS_constraints,
                                                NS_fe.component_mask(velocities));

	  VectorTools::interpolate_boundary_values(NS_dof_handler, 10, Functions::ZeroFunction<dim>(dim+1), nonzero_NS_constraints, NS_fe.component_mask(vertical_velocity));
      //VectorTools::interpolate_boundary_values(NS_dof_handler, 10, BoundaryValues<dim>(), nonzero_NS_constraints, NS_fe.component_mask(velocities));

      VectorTools::interpolate_boundary_values(NS_dof_handler,
                                               11, // Outlet
											   BoundaryValues<dim>(),//Functions::ZeroFunction<dim>(dim+1),
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
                                                3,
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

    std::cout << "Number of degrees of freedom: " << NS_dof_handler.n_dofs()
              << " (" << dof_u << " + " << dof_p << ')' << std::endl;
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
  void Problem<dim>::assemble_navier_stokes(const bool initial_step)
  {
    NS_system_matrix = 0; // It is technically unnecessary to assemble every time...
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

    auto ion_cell = dof_handler.begin_active();
    const auto ion_endc = dof_handler.end();
    std::vector<types::global_dof_index> ion_local_dof_indices(4);

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

        if (ion_cell != ion_endc)
        	ion_cell->get_dof_indices(ion_local_dof_indices);
        else
        	cout << "Warning! Reached end of ion cells at NS cell " << cell->index() << endl;

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

                if (ion_cell != ion_endc && i < 12) {
                	double E_x,E_y,ions;
					E_x = Field_X(ion_local_dof_indices[i % 3]);
					E_y = Field_Y(ion_local_dof_indices[i % 3]);
					ions = ion_density(ion_local_dof_indices[i % 3]);
                    f[0] = q0 * E_x / rho * ions;
                    f[1] = q0 * E_y / rho * ions;
                }

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

        const AffineConstraints<double> &constraints_used = initial_step ? nonzero_NS_constraints : zero_NS_constraints;
        constraints_used.distribute_local_to_global(local_matrix, local_rhs, local_dof_indices, NS_system_matrix, NS_system_rhs);

        if (ion_cell != ion_endc)
        	ion_cell++;

      }
        // Move the pressure mass matrix into a separate matrix:
        pressure_mass_matrix.reinit(NS_sparsity_pattern.block(1, 1));
        pressure_mass_matrix.copy_from(NS_system_matrix.block(1, 1));

        NS_system_matrix.block(1, 1) = 0;

  }

template <int dim>
void Problem<dim>::solve_nonlinear_navier_stokes_step(const bool initial_step)
{
  const AffineConstraints<double> &constraints_used =
    initial_step ? nonzero_NS_constraints : zero_NS_constraints;

  const double tol = 1e-4 * NS_system_rhs.block(0).linfty_norm(); // Changed from: const double tol = 1e-4 * NS_system_rhs.l2_norm();

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
void Problem<dim>::navier_stokes_newton_iteration( const double tolerance,const unsigned int max_n_line_searches)
{
      unsigned int line_search_n = 0;
      double       current_res   = 1.0 + tolerance;

      while ((current_res > tolerance) && line_search_n < max_n_line_searches)
        {
			assemble_navier_stokes(false);
			solve_nonlinear_navier_stokes_step(false);

			const double alpha = 1.;
			NS_solution.add(alpha, NS_newton_update);
			nonzero_NS_constraints.distribute(NS_solution);
			current_res = NS_newton_update.block(0).linfty_norm();
			std::cout << "  Residual: " << current_res  << " for " << line_search_n << " line searches" << std::endl;

			++line_search_n;
		  }
      if (line_search_n >= max_n_line_searches)
      	cout << "WARNING! NS achieved a residual " << current_res << " after " << line_search_n << " iterations" << endl;
}

template <int dim>
void Problem<dim>::solve_navier_stokes()
{
	evaluate_electric_field();

	std::cout << "Solving Navier Stokes... " << std::endl;

	if (step_number == 1) {
		assemble_navier_stokes(true);
		solve_nonlinear_navier_stokes_step(true);
		NS_solution = NS_newton_update;
		nonzero_NS_constraints.distribute(NS_solution);
		double current_res = NS_newton_update.block(0).linfty_norm();
		std::cout << "The residual of  the initial guess is " << current_res << std::endl;
	}

	std::cout << "Starting newton iteration for the NS... " << std::endl;
	const double tol = 1.e-6;
	const unsigned int max_it = 15;
	navier_stokes_newton_iteration(tol,max_it);

	cout << "Recovering velocity and pressure values for output... " << endl;

	Vel_X.reinit(dof_handler.n_dofs());
	Vel_Y.reinit(dof_handler.n_dofs());
	pressure.reinit(dof_handler.n_dofs());

	const unsigned int dofs_per_cell = 4;
	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
	const unsigned int dofs_per_NS_cell = 22;
	std::vector<types::global_dof_index> NS_local_dof_indices(dofs_per_NS_cell);

	auto cell = dof_handler.begin_active();
	auto NS_cell = NS_dof_handler.begin_active();

	const auto endc = dof_handler.end();
	const auto NS_endc = NS_dof_handler.end();

	double vel_max = 0.;

	while (cell != endc && NS_cell != NS_endc) {

		cell->get_dof_indices(local_dof_indices);
		NS_cell->get_dof_indices(NS_local_dof_indices);

		for (unsigned int k = 0; k < dofs_per_cell; ++k) {

			const unsigned int ind = local_dof_indices[k];

			Vel_X(ind) = NS_solution[NS_local_dof_indices[3*k]]; // Not so sure about this...
			Vel_Y(ind) = NS_solution[NS_local_dof_indices[3*k+1]]; // ... or this...
			pressure(ind) = NS_solution[NS_local_dof_indices[3*k+2]]; // ... or even this
			// But they all seem to work in the output!

			vel_max = std::max(vel_max,Vel_X(ind));

		}
		++cell;
		++NS_cell;
	}

	cout << "Estimating thrust..." << endl; estimate_thrust();
}

template <int dim>
 void Problem<dim>::estimate_thrust()
 {
   double integral = 0.;
   double vel_integral = 0.;

   double in_vel_integral = 0.;
   double out_vel_integral = 0.;

   double in_out_contribution = 0.;
   double wire_contribution = 0.;
   double naca_contribution = 0.;

   const unsigned int dofs_per_cell = NS_fe.n_dofs_per_cell();

   std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

   QGauss<dim-1> face_quadrature(degree+1);
   FEFaceValues<dim> fe_face_values(NS_fe, face_quadrature, update_values | update_normal_vectors | update_quadrature_points | update_JxW_values);

   const FEValuesExtractors::Vector velocities(0);
   const FEValuesExtractors::Scalar pressure(dim);

   std::vector<Tensor<1,dim>> 	velocity_values(face_quadrature.size());
   std::vector<double> 			pressure_values(face_quadrature.size());

   double value, vel_value, vel_sqr_value;

     for (const auto & cell : NS_dof_handler.active_cell_iterators()) {
  	   if (cell->at_boundary()) {
			 for (const auto face_index : GeometryInfo<dim>::face_indices()) {
			   if (cell->face(face_index)->at_boundary()) {

				   const double ID = cell->face(face_index)->boundary_id();

				   if ( ID != 0) {
						fe_face_values.reinit(cell, face_index);
						fe_face_values[velocities].get_function_values(NS_solution, velocity_values);
						fe_face_values[pressure].get_function_values(NS_solution, pressure_values);

						 for (const auto q_point : fe_face_values.quadrature_point_indices()) {

							 value = 0.;
							 vel_value = 0.;
							 vel_sqr_value = 0.;

							 //const Tensor<1,dim> n_q = fe_face_values.normal_vector(q_point);
							 //if ( std::fabs(std::sqrt(n_q[0]*n_q[0]+n_q[1]*n_q[1])-1.) >= 1.e-10)  cout << "WARNING! normal norm is " << std::sqrt(n_q[0]*n_q[0]+n_q[1]*n_q[1])  << endl;

							 Tensor<1,dim> n_q;
							 n_q[0] = 0.;
							 n_q[1] = 0.;

							 if (ID == 10)
								 n_q[0] = -1.;
							 else if (ID == 11)
								 n_q[0] = 1.;
							 else if (ID == 1)
								 n_q = get_emitter_normal(fe_face_values.quadrature_point(q_point)); // emitter normal pointing into the mesh
							 else if (ID == 2 || ID == 3)
								 n_q = -fe_face_values.normal_vector(q_point); // collector normal pointing into the mesh

							 if (ID == 10 || ID == 11) {
								   const double v_dot_n_q = velocity_values[q_point] * n_q;
								   vel_value = v_dot_n_q * fe_face_values.JxW(q_point);
								   vel_sqr_value = (v_dot_n_q * v_dot_n_q) * fe_face_values.JxW(q_point);
							 }
						   value = vel_sqr_value * ( (ID == 11) - (ID == 10) )
								   + pressure_values[q_point] * n_q[0] * fe_face_values.JxW(q_point);

						 } // cycle on quadrature points

						   integral += value;
						   vel_integral += vel_sqr_value* ( (ID == 11) - (ID == 10) );

						   if (ID == 1)
							   wire_contribution += value;
						   else if (ID == 10) {
							   in_vel_integral -= vel_value;
							   in_out_contribution += value;
						   } else if (ID == 11) {
							   out_vel_integral += vel_value;
							   in_out_contribution += value;
						   } else if (ID == 2 || ID == 3)
							   naca_contribution += value;

				   } // if ID != 0
			   } // if face at boundary
			 } // end cycle on faces
  	   } // if cell at boundary
     } // end cycle on cells

   cout << "Inlet velocity integral is " << in_vel_integral << " [m^2/s] while at the outlet it is " << out_vel_integral << " [m^2/s]" << endl;
   cout << "Velocity thrust contribution is " << vel_integral*rho << "[N/m]" << endl;

   const double thrust = integral * rho;

   cout << "Thrust per (half) foil span is " << thrust << " [N/m] " << endl;
   cout << " 	from: " << in_out_contribution/std::fabs(integral)*100. << " % edges, " << wire_contribution/std::fabs(integral)*100.  << " % emitter, " << naca_contribution/std::fabs(integral)*100.  << " % collector" << endl;
 }


template <int dim>
void Problem<dim>::refine_mesh()
{
	cout << "Executing mesh refinement... " << endl;
  Vector<float> error_estimated(triangulation.n_active_cells());

  KellyErrorEstimator<dim>::estimate(dof_handler,
            QGauss<dim - 1>(fe.degree + 1),
            {},
            ion_density,
            error_estimated);

  GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                      error_estimated,
                                                      0.02,
                                                      0.02);

  // Solution transfer
  triangulation.prepare_coarsening_and_refinement();

  SolutionTransfer<dim,Vector<double>> ion_transfer(dof_handler);
  ion_transfer.prepare_for_coarsening_and_refinement(ion_density);
  Vector<double> previous_ion(ion_density);

  SolutionTransfer<dim,Vector<double>> old_ion_transfer(dof_handler);
  old_ion_transfer.prepare_for_coarsening_and_refinement(old_ion_density);
  Vector<double> previous_old_ion(old_ion_density);

  SolutionTransfer<dim,Vector<double>> potential_transfer(dof_handler);
  potential_transfer.prepare_for_coarsening_and_refinement(potential);
  Vector<double> previous_potential(potential);

  SolutionTransfer<dim,BlockVector<double>> NS_transfer(NS_dof_handler);
  NS_transfer.prepare_for_coarsening_and_refinement(NS_solution);
  BlockVector<double> previous_NS(NS_solution);

  triangulation.execute_coarsening_and_refinement();

  setup_poisson();
  potential_transfer.interpolate(previous_potential, potential);
  constraints_poisson.distribute(potential);

  setup_drift_diffusion(/*re-initialize densities = */ true);
  setup_navier_stokes();

  ion_transfer.interpolate(previous_ion, ion_density);
  ion_constraints.distribute(ion_density);

  old_ion_transfer.interpolate(previous_old_ion, old_ion_density);
  ion_constraints.distribute(old_ion_density);

  NS_transfer.interpolate(previous_NS, NS_solution);
  nonzero_NS_constraints.distribute(NS_solution);
}


template <int dim>
void Problem<dim>::run()
{
	cout << "From Peek's law:  " << endl;
	cout << "	the field at emitter surface is " << Ep*1e-6 << " [MV/m] " <<endl;
	cout << "	the ionization radius is "<< Ri*1e+3 << " [mm]" << endl;
	cout << " 	the ionization potential is " << Vi*1e-3  << " [kV]" << endl;
	cout << endl;

    create_mesh();

    const unsigned int max_steps = 500;

	step_number = 0;
	setup_poisson();
	solve_homogeneous_poisson();

	setup_drift_diffusion(/*re-initialize densities = */ true);
	VectorTools::interpolate(mapping,dof_handler, Functions::ConstantFunction<dim>(N_0), old_ion_density);
	setup_navier_stokes();

	output_results(0);

	const double tol = 1.e-9;
	const unsigned int max_it = 1e+3;

	const double time_tol = 5.e-3; //1.e-3;
	double time_err = 1. + time_tol;

	while (step_number < max_steps && time_err > time_tol)
	  {

		++step_number;

		// Faster time-stepping (adaptive time-stepping would be MUCH better!)
		if (step_number % 40 == 1 && step_number > 1 && timestep < 1.e-3)
			timestep*= 10.;

		const double gummel_tol = 1.e-4;
		double err = gummel_tol + 1.;
		unsigned int it = 0;

		eta = old_ion_density;
		Vector<double> previous_density(old_ion_density.size());

		while (err > gummel_tol && it < max_it) {

			solve_nonlinear_poisson(tol, max_it); // Updates potential and eta

			previous_density = ion_density;
			perform_drift_diffusion_fixed_point_iteration_step(); // Updates ion_density
			previous_density -= ion_density;
			err = previous_density.linfty_norm()/ion_density.linfty_norm();

			eta = ion_density;
			it++;
		}
		if (it >= max_it)
			cout << "WARNING! DD achieved a relative error " << err << " after " << it << " iterations" << endl;

		previous_density = old_ion_density;
		previous_density -= ion_density;
		time_err = previous_density.linfty_norm()/old_ion_density.linfty_norm();
		cout << "Density change from previous time-step is: " << time_err*100. << " %" << endl;

		old_ion_density = ion_density;

		if (step_number % 40 == 1)  // Mesh refinement every 40 timesteps
			refine_mesh();

		if ( step_number % 40 == 1) // NS solution update every 40 timesteps
			solve_navier_stokes();

		output_results(step_number);

	  }

	std::cout << " 	Elapsed CPU time: " << timer.cpu_time()/60. << " minutes.\n" << std::endl << std::endl;

}



int main()
{
  try
    {
      Problem<2> drift_diffusion;
      drift_diffusion.run();
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

      return 1; // Report an error
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
