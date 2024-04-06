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

// This code is the original code written by Matteo Menessini modified in order to achieve a better readability and performances 
// by Tommaso Andena and Giacomo Bottacini (PACS project Politecnico di Milano a.e 2023/2024)

// This code solve the Dirift Diffusion equation in a rectangular pn junction, in particular we perform a validation on the solver for
// the electrical part of the problem, we test the method considering a semiconductor problem, whit a well known solution.
// The problem consists in solving for the potential, charge and hole densities.
// This code contains only the main and some include (for now)

// Time-stepping from step-26 deal ii tutorial

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

#include "Problem.hpp" //header file that contains the template class Problem created by MatMes. It also contains Electrical_Values/Constants


using namespace dealii;  //entrambi inutili
using namespace std;


// METODO COMMENATTO DA MENESSINI NON è OPERA NOSTRA 
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
