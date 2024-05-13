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

using namespace dealii;
using namespace std;

// Physical Constants
const double eps_0 = 8.854 * 1.e-12; //[F/m]= [C^2 s^2 / kg / m^3]
const double eps_r = 4.;
const double q0 = 1.602 * 1.e-19; // [C]
const double kB = 1.381 * 1.e-23 ; //[J/K]

const double q_over_eps_0 = q0 / eps_0; // [m^3 kg C^-1 s^-2]
const double L = 1.5e-6;
const double A = 1.e+22;
const double E = 1.e+22;

const double N_0 = 1.e+16;

const double N1 = E/2. + std::sqrt(E*E+4.*N_0*N_0)/2.; // [m^-3] Electron density on boundary 1
const double P2 = A/2. + std::sqrt(A*A+4.*N_0*N_0)/2.; // [m^-3] Electron density on boundary 1
const double P1 = N_0*N_0/N1; // [m^-3] Electron density on boundary 2
const double N2 = N_0*N_0/P2; // [m^-3] Electron density on boundary 2
const double mup = 1.e-1; // [m^2/s/V]
const double mun = 3.e-2; // [m^2/s/V]
const double V_E = 26e-3; // [V] ion temperature

const double Dp = mup * V_E;
const double Dn = mun * V_E;


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


FullMatrix<double> compute_triangle_matrix(const Point<2> a, const Point<2> b, const Point<2> c, const double alpha12, const double alpha23, const double alpha31, const double D)
{
	const unsigned int size = 3;
	FullMatrix<double> tria_matrix(size,size);

	tria_matrix = 0;

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

  /*template <int dim>
   class ExactPotentialValues : public Function<dim>
   {
   public:
 	 ExactPotentialValues() : Function<dim>()
     {}

     virtual double value(const Point<dim> & p, const unsigned int component = 0) const override;

   };

  template <int dim>
  double ExactPotentialValues<dim>::value(const Point<dim> & p,
                                   const unsigned int component) const
  {
    (void)component;
    AssertIndexRange(component, 1);
    Assert(dim == 2, ExcNotImplemented());

    const double w = 1.261e-7;
    const double q = 2.2589e+13;
    const double phi0 = 0.3592;

    const double w_left = L/2. - w;
    const double w_right =  L/2. + w;


   if (p[0] < w_left)
    	return V_E*std::log(N1/N_0);
   else if (p[0] < L/2. && p[0] >= w_left)
   	   return phi0 - q*pow(p[0]-w_left,2.);
    else if (p[0] > w_right)
    	return V_E*std::log(N2/N_0);
    else if (p[0] >= L/2. && p[0] < w_right)
    	return - phi0 + q*pow(p[0]-w_right,2.);
    else
    	Assert(false, ExcNotImplemented());
  }*/

  template <int dim>
     class DopingValues : public Function<dim>
     {
     public:
   	 DopingValues() : Function<dim>()
       {}

       virtual double value(const Point<dim> & p, const unsigned int component = 0) const override;

     };

    template <int dim>
    double DopingValues<dim>::value(const Point<dim> & p,
                                     const unsigned int component) const
    {
      (void)component;
      AssertIndexRange(component, 1);
      Assert(dim == 2, ExcNotImplemented());

      if (p[0] < 0.5*L)
      	return E;
      else
      	return -A;
    }

    template <int dim>
     class PotentialValues : public Function<dim>
     {
     public:
   	 PotentialValues() : Function<dim>()
       {}

       virtual double value(const Point<dim> & p, const unsigned int component = 0) const override;

     };

    template <int dim>
    double PotentialValues<dim>::value(const Point<dim> & p,
                                     const unsigned int component) const
    {
      (void)component;
      AssertIndexRange(component, 1);
      Assert(dim == 2, ExcNotImplemented());

     if (p[0] <= 0.45*L)
      	return V_E*std::log(N1/N_0);
      else
      	return V_E*std::log(N2/N_0);
    }

    template <int dim>
     class ElectronInitialValues : public Function<dim>
     {
     public:
   	 ElectronInitialValues() : Function<dim>()
       {}

       virtual double value(const Point<dim> & p, const unsigned int component = 0) const override;

     };

    template <int dim>
    double ElectronInitialValues<dim>::value(const Point<dim> & p,
                                     const unsigned int component) const
    {
      (void)component;
      AssertIndexRange(component, 1);
      Assert(dim == 2, ExcNotImplemented());

      if (p[0] <= 0.45*L)
      	return N1;
      else
      	return N2;
    }


    template <int dim>
     class IonInitialValues : public Function<dim>
     {
     public:
   	 IonInitialValues() : Function<dim>()
       {}

       virtual double value(const Point<dim> & p, const unsigned int component = 0) const override;

     };

    template <int dim>
    double IonInitialValues<dim>::value(const Point<dim> & p,
                                     const unsigned int component) const
    {
      (void)component;
      AssertIndexRange(component, 1);
      Assert(dim == 2, ExcNotImplemented());

      if (p[0] <= 0.55*L)
      	return P1;
      else
      	return P2;
    }

template <int dim>
Problem<dim>::Problem()
  : fe(1) // linear elements
  , dof_handler(triangulation)
  , step_number(0)
  , mapping()
{}


template <int dim>
void Problem<dim>::create_mesh()
{
    const Point<dim> bottom_left(0.,-L/20.);
	const Point<dim> top_right(L,L/20.);

	// For a structured mesh
	//GridGenerator::subdivided_hyper_rectangle(triangulation, {100,}, bottom_left, top_right);

	const std::string filename = "../../../Mesh_Menessini/small_square.msh";
		ifstream input_file(filename);
		cout << "Reading from " << filename << endl;
	    GridIn<2>       grid_in;
	    grid_in.attach_triangulation(triangulation);
	    grid_in.read_msh(input_file);

	  for (auto &face : triangulation.active_face_iterators())
	  {
		  if (face->at_boundary())
		  {
			  face->set_boundary_id(0);
			  const Point<dim> c = face->center();

			  	  if ( c[1] < top_right[1] && c[1] > bottom_left[1]) {

			  		  if (c[0] < (top_right[0] + bottom_left[0])/2.) {
			  			  face->set_boundary_id(1);
			  		  } else
			  			face->set_boundary_id(2);
			  	  }

			  }
		}

	 triangulation.refine_global(3);
}


template <int dim>
void Problem<dim>::setup_poisson()
{
	dof_handler.distribute_dofs(fe);

	potential.reinit(dof_handler.n_dofs());
	poisson_newton_update.reinit(dof_handler.n_dofs());

	constraints_poisson.clear();
	DoFTools::make_hanging_node_constraints(dof_handler, constraints_poisson);
	VectorTools::interpolate_boundary_values(dof_handler, 1, Functions::ZeroFunction<dim>(), constraints_poisson);
	VectorTools::interpolate_boundary_values(dof_handler, 2, Functions::ZeroFunction<dim>(), constraints_poisson);
	constraints_poisson.close();

	// Used for the update term in Newton's method
	zero_constraints_poisson.clear();
	DoFTools::make_hanging_node_constraints(dof_handler, zero_constraints_poisson);
	VectorTools::interpolate_boundary_values(dof_handler, 1, Functions::ZeroFunction<dim>(), zero_constraints_poisson);
	VectorTools::interpolate_boundary_values(dof_handler, 2, Functions::ZeroFunction<dim>(), zero_constraints_poisson);
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

  for (unsigned int i = 0; i < old_ion_density.size(); ++i)
	  ion_mass_matrix(i,i) = mass_matrix_poisson(i,i) * (old_ion_density(i) + old_electron_density(i));

  system_matrix_poisson.add(q0 / V_E, ion_mass_matrix);
  cout << "Ion matrix norm is " << system_matrix_poisson.linfty_norm() << endl;
  system_matrix_poisson.add(eps_r * eps_0, laplace_matrix_poisson);

  cout << "Matrix norm is " << system_matrix_poisson.linfty_norm() << endl;

  // Assmble RHS
  poisson_rhs = 0;
    Vector<double> tmp(dof_handler.n_dofs());
    Vector<double> doping_and_ions(dof_handler.n_dofs());
    VectorTools::interpolate(mapping,dof_handler, DopingValues<dim>(), doping_and_ions);

    doping_and_ions -= old_electron_density;
    doping_and_ions += old_ion_density;

    mass_matrix_poisson.vmult(tmp,doping_and_ions);
    poisson_rhs.add(q0, tmp);//0, tmp);//
    laplace_matrix_poisson.vmult(tmp,potential);
    poisson_rhs.add(- eps_r * eps_0, tmp);//- eps_r * eps_0, tmp);//

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
void Problem<dim>::newton_iteration_poisson(const double tolerance, const unsigned int max_n_line_searches)
  {
	unsigned int line_search_n = 1;
	double current_res =  tolerance + 1;

	while (current_res > tolerance && line_search_n <= max_n_line_searches)
	  {
			assemble_nonlinear_poisson();
			solve_poisson();

			// Update Clamping
			const double alpha = 1.;
			cout << "Norm before clamping is " << poisson_newton_update.linfty_norm() << endl;
			for (unsigned int i = 0; i < poisson_newton_update.size(); i++) {
				poisson_newton_update(i) = std::max(std::min(poisson_newton_update(i),V_E),-V_E);

				old_electron_density(i) *= std::exp(alpha*poisson_newton_update(i)/V_E);
				old_ion_density(i) *= std::exp(-alpha*poisson_newton_update(i)/V_E);
			}
			constraints.distribute(old_ion_density);
			constraints.distribute(old_electron_density);

			potential.add(alpha, poisson_newton_update);
			constraints_poisson.distribute(potential);

			current_res = poisson_newton_update.linfty_norm();

			std::cout << "  alpha: " << std::setw(10) << alpha  << std::setw(0) << "  residual: " << current_res  << std::endl;
			std::cout << "  number of line searches: " << line_search_n << "  residual: " << current_res << std::endl;

			++line_search_n;
			//output_results(step_number); // Only needed to see the update at each step during testing
	  }
  }


template <int dim>
void Problem<dim>::setup_drift_diffusion()
{
	ion_density.reinit(dof_handler.n_dofs());
	electron_density.reinit(dof_handler.n_dofs());

	old_ion_density.reinit(dof_handler.n_dofs());
	old_electron_density.reinit(dof_handler.n_dofs());

	constraints.clear();
	DoFTools::make_hanging_node_constraints(dof_handler, constraints);
	constraints.close();

	DynamicSparsityPattern dsp(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
	sparsity_pattern.copy_from(dsp);

	ion_rhs.reinit(dof_handler.n_dofs());
	electron_rhs.reinit(dof_handler.n_dofs());

	ion_system_matrix.reinit(sparsity_pattern);
	electron_system_matrix.reinit(sparsity_pattern);

	drift_diffusion_matrix.reinit(sparsity_pattern);
	electron_drift_diffusion_matrix.reinit(sparsity_pattern);
}


template <int dim>
void Problem<dim>::assemble_drift_diffusion_matrix()
{
	electron_rhs = 0;
	ion_rhs = 0;
	drift_diffusion_matrix = 0;
	electron_drift_diffusion_matrix = 0;

  const unsigned int vertices_per_cell = 4;
  std::vector<types::global_dof_index> local_dof_indices(vertices_per_cell);

  const unsigned int t_size = 3;
  Vector<double> cell_rhs(t_size);
  FullMatrix<double> A(t_size,t_size), B(t_size,t_size), neg_A(t_size,t_size), neg_B(t_size,t_size);
  std::vector<types::global_dof_index> A_local_dof_indices(t_size);
  std::vector<types::global_dof_index> B_local_dof_indices(t_size);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
	    A = 0;
	    B = 0;
	    neg_A = 0;
		neg_B = 0;
		cell_rhs = 0;
		cell->get_dof_indices(local_dof_indices);

		// Lexicographic ordering
		const Point<dim> v1 = cell->vertex(2); // top left
		const Point<dim> v2 = cell->vertex(3); // top right
		const Point<dim> v3 = cell->vertex(0); // bottom left
		const Point<dim> v4 = cell->vertex(1); // bottom right

		const double u1 = -potential[local_dof_indices[2]]/V_E;
		const double u2 = -potential[local_dof_indices[3]]/V_E;
		const double u3 = -potential[local_dof_indices[0]]/V_E;
		const double u4 = -potential[local_dof_indices[1]]/V_E;

		const double l_alpha = side_length(v1,v4);
		const double l_beta = side_length(v2,v3);

		const double alpha21 = (u1 - u2);
		const double alpha42 = (u2 - u4);
		const double alpha34 = (u4 - u3);
		const double alpha13 = (u3 - u1);

		const double neg_alpha21 =  - (u1 - u2);
		const double neg_alpha42 =  - (u2 - u4);
		const double neg_alpha34 = - (u4 - u3);
		const double neg_alpha13 = - (u3 - u1);

		if (l_alpha >= l_beta) { // l_alpha is the longest diagonal: split by beta
					const double alpha23 =  (u3 - u2);
					const double neg_alpha23 = - (u3 - u2);
					//cout << "Alpha 23 is: " << alpha23 << endl;

					// Triangle A:
					A= compute_triangle_matrix(v2,v1,v3, alpha21, alpha13, -alpha23, Dp);
					neg_A= compute_triangle_matrix(v2,v1,v3, neg_alpha21, neg_alpha13, -neg_alpha23, Dn);

					// Triangle B:
					B = compute_triangle_matrix(v3,v4,v2, alpha34, alpha42, alpha23, Dp);
					neg_B = compute_triangle_matrix(v3,v4,v2, neg_alpha34, neg_alpha42, neg_alpha23, Dn);

					// Matrix assemble
					A_local_dof_indices[0] = local_dof_indices[3];
					A_local_dof_indices[1] = local_dof_indices[2];
					A_local_dof_indices[2] = local_dof_indices[0];

					B_local_dof_indices[0] = local_dof_indices[0];
					B_local_dof_indices[1] = local_dof_indices[1];
					B_local_dof_indices[2] = local_dof_indices[3];

				} else { // l_beta is the longest diagonal: split by alpha
					const double alpha14 = (u4 - u1);
					const double neg_alpha14 = - (u4 - u1);
					//cout << "Alpha 14 is: " << alpha14 << endl;

					// Triangle A:
					A = compute_triangle_matrix(v4,v2,v1, alpha42, alpha21, alpha14, Dp);
					neg_A = compute_triangle_matrix(v4,v2,v1, neg_alpha42, neg_alpha21, neg_alpha14, Dn);

					// Triangle B:
					B = compute_triangle_matrix(v1,v3,v4, alpha13, alpha34, -alpha14, Dp);
					neg_B = compute_triangle_matrix(v1,v3,v4, neg_alpha13, neg_alpha34, -neg_alpha14, Dn);

					A_local_dof_indices[0] = local_dof_indices[1];
					A_local_dof_indices[1] = local_dof_indices[3];
					A_local_dof_indices[2] = local_dof_indices[2];

					B_local_dof_indices[0] = local_dof_indices[2];
					B_local_dof_indices[1] = local_dof_indices[0];
					B_local_dof_indices[2] = local_dof_indices[1];

				}

				constraints.distribute_local_to_global(A, cell_rhs,  A_local_dof_indices, drift_diffusion_matrix,ion_rhs);
				constraints.distribute_local_to_global(B, cell_rhs,  B_local_dof_indices, drift_diffusion_matrix, ion_rhs);

				constraints.distribute_local_to_global(neg_A, cell_rhs,  A_local_dof_indices, electron_drift_diffusion_matrix, electron_rhs);
				constraints.distribute_local_to_global(neg_B, cell_rhs,  B_local_dof_indices, electron_drift_diffusion_matrix, electron_rhs);
		    }


}

template <int dim>
void Problem<dim>::apply_drift_diffusion_boundary_conditions()
{
	    std::map<types::global_dof_index, double> emitter_boundary_values, collector_boundary_values;

		VectorTools::interpolate_boundary_values(mapping, dof_handler,1, Functions::ConstantFunction<dim>(N1), emitter_boundary_values);
	    MatrixTools::apply_boundary_values(emitter_boundary_values, electron_system_matrix, electron_density, electron_rhs);

		VectorTools::interpolate_boundary_values(mapping, dof_handler,2, Functions::ConstantFunction<dim>(N2), collector_boundary_values);
		MatrixTools::apply_boundary_values(collector_boundary_values, electron_system_matrix, electron_density, electron_rhs);

		VectorTools::interpolate_boundary_values(mapping, dof_handler,1, Functions::ConstantFunction<dim>(P1), emitter_boundary_values);
		MatrixTools::apply_boundary_values(emitter_boundary_values, ion_system_matrix, ion_density, ion_rhs);

		VectorTools::interpolate_boundary_values(mapping, dof_handler,2, Functions::ConstantFunction<dim>(P2), collector_boundary_values);
		MatrixTools::apply_boundary_values(collector_boundary_values, ion_system_matrix, ion_density, ion_rhs);
 }

template <int dim>
void Problem<dim>::solve_drift_diffusion()
{
  SparseDirectUMFPACK P_direct;
  P_direct.initialize(ion_system_matrix);
  P_direct.vmult(ion_density, ion_rhs);
  constraints.distribute(ion_density);

  SparseDirectUMFPACK N_direct;
  N_direct.initialize(electron_system_matrix);
  N_direct.vmult(electron_density, electron_rhs);
  constraints.distribute(electron_density);
}



template <int dim>
void Problem<dim>::output_results(const unsigned int step)
{
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    data_out.add_data_vector(old_ion_density, "Old_Ion_Density");
    data_out.add_data_vector(old_electron_density, "Old_Electron_Density");
    data_out.add_data_vector(ion_density, "Ion_Density");
    data_out.add_data_vector(electron_density, "Electron_Density");
    data_out.add_data_vector(potential, "Potential");
    data_out.build_patches();

    std::string filename;
    filename = "solution-" + Utilities::int_to_string(step, 3) + ".vtk";
    DataOutBase::VtkFlags vtk_flags;
    vtk_flags.compression_level = DataOutBase::VtkFlags::ZlibCompressionLevel::best_speed;
    data_out.set_flags(vtk_flags);
    std::ofstream output(filename);
    data_out.write_vtk(output);
}

template <int dim>
void Problem<dim>::run()
{
	create_mesh();

	setup_poisson();
	setup_drift_diffusion();

	VectorTools::interpolate(mapping, dof_handler, PotentialValues<dim>(), potential);
  constraints_poisson.distribute(potential);
	VectorTools::interpolate(mapping, dof_handler, IonInitialValues<dim>(), old_ion_density);
	VectorTools::interpolate(mapping, dof_handler, ElectronInitialValues<dim>(), old_electron_density);

    output_results(0);

    Vector<double> tmp(ion_density.size());

    const double tol = 1.e-9*V_E;
    const unsigned int max_it = 50;

    double ion_tol = 1.e-10;
    double electron_tol = 1.e-10;

    double ion_err = ion_tol + 1.;
    double electron_err = electron_tol + 1.;

    while ( (ion_err > ion_tol || electron_err > electron_tol) && step_number < 10)  //time <= max_time - 0.1*timestep
      {
        ++step_number;

        // Solve Non-Linear Poisson
		newton_iteration_poisson(tol, max_it);

		//VectorTools::interpolate(mapping, dof_handler, ExactPotentialValues<dim>(), potential);

		// Drift Diffusion Step
        assemble_drift_diffusion_matrix();

		ion_system_matrix.copy_from(drift_diffusion_matrix);
		electron_system_matrix.copy_from(electron_drift_diffusion_matrix);

		// BE integration
        /*mass_matrix.vmult(ion_rhs, old_ion_density);
        mass_matrix.vmult(electron_rhs, old_electron_density);
        assemble_drift_diffusion_matrix();

        RightHandSide<dim> rhs_function;
        rhs_function.set_time(time);
        VectorTools::create_right_hand_side(dof_handler, QGauss<dim>(fe.degree + 1), rhs_function, tmp);
        forcing_terms = tmp;
        forcing_terms *= timestep;

        ion_rhs += forcing_terms;
        ion_system_matrix.copy_from(mass_matrix);
        ion_system_matrix.add(timestep, drift_diffusion_matrix);

        electron_rhs += forcing_terms;
        electron_system_matrix.copy_from(mass_matrix);
        electron_system_matrix.add(timestep, negative_drift_diffusion_matrix);*/

        apply_drift_diffusion_boundary_conditions();
        solve_drift_diffusion();

        // Update error for convergence
        electron_tol = 1.e-10*old_electron_density.linfty_norm();
        ion_tol = 1.e-10*old_ion_density.linfty_norm();

        tmp = ion_density;
        tmp -= old_ion_density;
        ion_err = tmp.linfty_norm();

        tmp = electron_density;
        tmp -= old_electron_density;
        electron_err = tmp.linfty_norm();

        output_results(step_number);

        old_ion_density = ion_density;
        old_electron_density = electron_density;

    	std::cout << " 	Elapsed CPU time: " << timer.cpu_time() << " seconds.\n" << std::endl << std::endl;
      }

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
