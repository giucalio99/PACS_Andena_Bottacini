/* -----------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2017 Jie Cheng
 *
 * This file is part of the deal.II code gallery.
 *
 * -----------------------------------------------------------------------------
 *
 * Author: Jie Cheng <chengjiehust@gmail.com>
 */

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

#include "InsIMEX.hpp"
#include "BlockSchurPreconditioner.hpp"



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
const double Re = 2.5e-5; // [m] emitter radius                                 !!!!
const double X = -Re; // [m] emitter center                                     !!!!

const double g = 0.02; // [m] interelectrode distance

// collector
const double collector_length = 0.1; // [m]                                     !!!!

// Peek's law (empyrical)
const double eps = 1.; // wire surface roughness correction coefficient
const double Ep = E_ON*delta*eps*(1+0.308/std::sqrt(Re*1.e+2*delta));
const double Ri = Ep/E_ON*Re; // [m] ionization radius
const double Vi = Ve - Ep*Re*log(Ep/E_ON); // [V] voltage on ionization region boundary

using namespace dealii;

//Must insert the part to import the mesh created in gmesh




// @sect3{Time stepping}
// This class is pretty much self-explanatory.
class Time
{
  public:
    Time(const double time_end,
         const double delta_t,
         const double output_interval,
         const double refinement_interval)
      : timestep(0),
        time_current(0.0),
        time_end(time_end),
        delta_t(delta_t),
        output_interval(output_interval),
        refinement_interval(refinement_interval)
    {
    }
    double current() const { return time_current; }
    double end() const { return time_end; }
    double get_delta_t() const { return delta_t; }
    unsigned int get_timestep() const { return timestep; }
    bool time_to_output() const;
    bool time_to_refine() const;
    void increment();

  private:
    unsigned int timestep;
    double time_current;
    const double time_end;
    const double delta_t;
    const double output_interval;
    const double refinement_interval;
};

bool Time::time_to_output() const
  {
    unsigned int delta = static_cast<unsigned int>(output_interval / delta_t);
    return (timestep >= delta && timestep % delta == 0);
  }

bool Time::time_to_refine() const
  {
    unsigned int delta = static_cast<unsigned int>(refinement_interval / delta_t);
    return (timestep >= delta && timestep % delta == 0);
  }

void Time::increment()
  {
    time_current += delta_t;
    ++timestep;
  }












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

  












void create_triangulation(Triangulation<2> &tria)
  {
    const std::string filename = "./Meshes/BOXED_ELLIPSE.msh";
    cout << "Reading from " << filename << endl;
    std::ifstream input_file(filename);
    GridIn<2>       grid_in;
    grid_in.attach_triangulation(triangulation);            //Attach this triangulation to be fed with the grid data
    grid_in.read_msh(input_file);                           //Read grid data from an msh file

    const types::manifold_id emitter = 1;                   //The type used to denote manifold indicators associated with every object of the mesh
    const Point<dim> center(X,0.);
    SphericalManifold<2> emitter_manifold(center);

    const types::manifold_id collector = 2;
    CollectorGeometry<2> collector_manifold;                

    tria.set_all_manifold_ids_on_boundary(1, emitter);
    tria.set_manifold(emitter, emitter_manifold);
    tria.set_all_manifold_ids_on_boundary(2, collector);
    tria.set_manifold(collector, collector_manifold);
    cout  << "Active cells: " << tria.n_active_cells() << endl;
  }


// Collector Manifold - START

double get_collector_height(const double &p)
{
  const double x = (X-g)/collector_length;
	double y = 0;

	if ( abs(x-1.) > 1e-12 && abs(x) > 1e-12 ) {
		double a0 = 0.2969;
		double a1 = -0.126;
		double a2 = -0.3516;
		double a3 = 0.2843;
		double a4 = -0.1036; // or -0.1015 for an open trailing edge
		double t = 0.1; // Last 2 digits of the NACA by 100

		y = 5*t*( a0 * sqrt(x) + a1 * x + a2 * pow(x,2.0) + a3 * pow(x,3.0) + a4 * pow(x,4.0) );
	}

	return y * collector_length;
}



template <int dim>
class CollectorGeometry : public ChartManifold<dim, dim, dim-1>     //ChartManifold is a class describes mappings that can be expressed in terms of charts.
  {
public:
  virtual Point<dim-1> pull_back(const Point<dim> &space_point) const override;        //Pull back the given point in spacedim to the Euclidean chartdim dimensional space

  virtual Point<dim> push_forward(const Point<dim-1> &chart_point) const override;     //Given a point in the chartdim dimensional Euclidean space, this method returns a point on the manifold embedded in the spacedim Euclidean space.

  virtual std::unique_ptr<Manifold<dim, dim>> clone() const override;                  //Return a copy of this manifold

  };

template <int dim>
std::unique_ptr<Manifold<dim, dim>> CollectorGeometry<dim>::clone() const
  {
return std::make_unique<CollectorGeometry<dim>>();
  }


template <int dim>
Point<dim> CollectorGeometry<dim>::push_forward(const Point<dim-1>  &x) const          //Input: a chart point that in our case is a 1D point 
  {
const double y = get_collector_height(x[0]);

Point<dim> p;
p[0] = x[0]; p[1] = y;

if (dim == 3) {
  p[2] = x[1];
}

return p;                                                                              //Output: a point of our collector in 2D 
  }


template <int dim>
Point<dim-1>  CollectorGeometry<dim>::pull_back(const Point<dim> &p) const             //Input: a point in our 2D mesh
{
  Point<dim-1> x;
  x[0] = p[0];

if (dim == 3) {
  x[1] = p[2];
}

return x;                                                                              //Output: a chart point that in our case is a 1D point
}  
// Collector Manifold - END


  


  

// @sect3{main function}
//
int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace fluid;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      parallel::distributed::Triangulation<2> tria(MPI_COMM_WORLD);
      create_triangulation(tria);
      InsIMEX<2> flow(tria);
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
