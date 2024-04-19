template <int dim>
std::unique_ptr<Manifold<dim, dim>> CollectorGeometry<dim>::clone() const
{
return std::make_unique<CollectorGeometry<dim>>();
}
//----------------------------------------------------------------------------------------------------------------------------------------------------

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

//-------------------------------------------------------------------------------------------------------------------------------------------------

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

//########################## HELPER FUNCTIONS #################################################################################################à
double get_collector_height(const double &p)
  {
  	if (p <= g || p >= g + collector_length)
  		return 0.;

  	const double a = collector_length/2.;
  	const double b = collector_height;

  	return b*std::sqrt(1.-(p-XC)/a*(p-XC)/a);
  }
//--------------------------------------------------------------------------------------------------------------------------------------------------
Tensor<1,2> get_emitter_normal(const Point<2> a) {   //non capisco dove la usa

	Tensor<1,2> normal;

	normal[0] = a[0] - X;
	normal[1] = a[1];

	const double norm = std::sqrt(normal[0]*normal[0]+normal[1]*normal[1]);

	return normal/norm;
}
//-----------------------------------------------------------------------------------------------------------------------------------------------
Tensor<1,2> get_collector_normal(const Point<2> a) {

	Tensor<1,2> normal;

	normal[0] = a[0] - XC;
	normal[1] = a[1];

	const double norm = std::sqrt(normal[0]*normal[0]+normal[1]*normal[1]);

	return normal/norm;
}