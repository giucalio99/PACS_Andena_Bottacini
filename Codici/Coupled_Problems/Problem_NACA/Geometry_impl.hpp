#include "Geometry.hpp"

template <int dim,int sign>
std::unique_ptr<Manifold<dim, dim>> CollectorGeometry<dim,sign>::clone() const
{
return std::make_unique<CollectorGeometry<dim,sign>>();
}

//--------------------------------------------------------------------------------------------------------------------------------------------

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

//------------------------------------------------------------------------------------------------------------------------------------------

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

// ############################ HELPER FUNCTIONS ##################################################################################


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

//-------------------------------------------------------------------------------------------------------------------------------------------------

Tensor<1,2> get_emitter_normal(const Point<2> a) {

	Tensor<1,2> normal;

	normal[0] = a[0] - X;
	normal[1] = a[1];

	const double norm = std::sqrt(normal[0]*normal[0]+normal[1]*normal[1]);

	return normal/norm;
}