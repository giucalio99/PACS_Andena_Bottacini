#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

using namespace dealii;
using namespace std;

#include<cmath>
#include<memory>

template <int dim,int sign>
class CollectorGeometry : public ChartManifold<dim, dim, dim-1>
{
public:
virtual Point<dim-1> pull_back(const Point<dim> &space_point) const override;

virtual Point<dim> push_forward(const Point<dim-1> &chart_point) const override;

virtual std::unique_ptr<Manifold<dim, dim>> clone() const override;

};

// HELPER FUNCTION
double get_collector_height(const double &p);
Tensor<1,2> get_emitter_normal(const Point<2> a);

#include"Geometry_impl.hpp"
#endif //GEOMETRY_HPP