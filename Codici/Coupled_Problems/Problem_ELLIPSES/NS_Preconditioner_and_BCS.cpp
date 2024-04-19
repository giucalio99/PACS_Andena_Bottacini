#ifndef "NS_PRECONDITIONER_AND_BCS"
#define "NS_PRECONDITIONER_AND_BCS"

// ############################# NS preconditioner ###############################################################################

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

// NS preconditioner implementation

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

// --------------------------------------------------------------------------------------------------------------------------------------

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

// ############################ Boundary Values #######################################################################################################

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

// Boundary Values implementations

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











#endif "NS_PRECONDITIONER_AND_BCS"