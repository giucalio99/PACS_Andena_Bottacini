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

// This code is the original code written by Matteo Menessini modified in order to achieve a better readability 
// by Tommaso Andena and Giacomo Bottacini

// This code solve the Dirift Diffusion equation in a rectangular pn junction, in particular we perform a validation on the solver for
// the electrical part of the problem, we test the method considering a semiconductor problem, whit a well known solution.
// The problem consists in solving for the potential, charge and hole densities.

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

#include "Problem.hpp" //header file that contains the template class Problem created by MM

using namespace dealii;
using namespace std;


//################################################# PHYSICAL CONSTANTS #############################################################################################################################

const double eps_0 = 8.854 * 1.e-12;    // [F/m]= [ (C^2 * s^2) / (kg * m^3)] = [ C^2 /(N * m^2)] Permittivity of free space
const double eps_r = 4.;                // [ADIM] Relative permittivity
const double q0 = 1.602 * 1.e-19;       // [C] Elementary charge
const double kB = 1.381 * 1.e-23 ;      // [J/K] Boltzmann constant

const double q_over_eps_0 = q0 / eps_0; // [m^3 kg C^-1 s^-2]
const double L = 1.5e-6;                // Length of the domain
const double A = 1.e+22;                // Negative doping value
const double E = 1.e+22;                // Positive doping value ( nella tesi è chiamato D (?))

const double N_0 = 1.e+16;              // constant starting ion density, charge density of ions in the ambient

const double N1 = E/2. + std::sqrt(E*E+4.*N_0*N_0)/2.; // [m^-3] Electron density on boundary 1
const double P2 = A/2. + std::sqrt(A*A+4.*N_0*N_0)/2.; // [m^-3] Electron density on boundary 1
const double P1 = N_0*N_0/N1; // [m^-3] Electron density on boundary 2
const double N2 = N_0*N_0/P2; // [m^-3] Electron density on boundary 2

const double mup = 1.e-1;     // [(m^2)/(s*V)] Mobility of positive charges 
const double mun = 3.e-2;     // [(m^2)/(s*V)] Mobility of holes
const double V_E = 2.6e-2;    // [V] ion temperature in Volts for both 

const double Dp = mup * V_E;  // Diffusion coefficient for holes (by Einstein relation)
const double Dn = mun * V_E;  // Diffusion coefficinet for electrons (by Einstein relation)

//#########################################################################################################################################################################################################


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

//################################### MAIN ###############################################################################################################################################

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
