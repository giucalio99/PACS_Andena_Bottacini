#ifndef BOUNDARIES_VALUES_HPP
#define BOUNDARIES_VALUES_HPP

using namespace dealii;
using namespace std;


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


// ----------------------------------- implementation -----------------------------------------------------------------------------------------



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

#endif // BOUNDARIES_VALUES_HPP