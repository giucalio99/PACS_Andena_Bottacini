using namespace dealii; 

// ---------------------------------------- DOPING VALUES --------------------------------------------------------------------------------------------------------

// this method return the value of the doping of the pn junction: if we are in the first half (x-axis) of the domain the doping in positive
// otherwise is negative

template <int dim>
double DopingValues<dim>::value(const Point<dim> & p, const unsigned int component) const{

    (void)component; //The function may not use the component parameter, so this line serves to silence any warnings about it being unused.
    AssertIndexRange(component, 1); //controllare il range di component, probabilmente controlla che la prima coordinata sia dentro il dominio(?)

    //assertion that checks whether the template parameter dim equals 2, if dim is not equal to 2, the assertion will fail, 
    //and an exception of type ExcNotImplemented() will be thrown. This function is only for 2D
    Assert(dim == 2, ExcNotImplemented()); 
    
    // check the x coordinate of the point
    if (p[0] < 0.5*L)
    return D;
    else
    return -A;

}

// --------------------------------------- POTENTIAL VALUES ---------------------------------------------------------------------------------------------------

template <int dim>
double PotentialValues<dim>::value(const Point<dim> & p, const unsigned int component) const{
    
    // these 3 lines are identical to the first template, same reasoning
    (void)component;
    AssertIndexRange(component, 1);
    Assert(dim == 2, ExcNotImplemented());

    if (p[0] <= 0.5*L) // check the x component of the point p
    return V_TH*std::log(D/ni);
    else
    return -V_TH*std::log(A/ni);

}

// --------------------------------------- ELECTRON INITIAL VALUES ---------------------------------------------------------------------------------------------

// This method return the value of the electron density, I think we initilaize the first 45% of the domain to N1 density and the other 55% to N2

template <int dim>
double ElectronInitialValues<dim>::value(const Point<dim> & p, const unsigned int component) const{
    
    // These 3 lines are identical to the first template, same resoning
    (void)component;
    AssertIndexRange(component, 1);
    Assert(dim == 2, ExcNotImplemented());
    
    //check the x component
    if (p[0] <= 0.45*L)
    return N1;
    else
    return N2;
}


// --------------------------------------- ION INITIAL VALUES -----------------------------------------------------------------------------------------------------


template <int dim>
double HoleInitialValues<dim>::value(const Point<dim> & p, const unsigned int component) const {
    
    // These 3 lines are identical to the first
    (void)component;
    AssertIndexRange(component, 1);
    Assert(dim == 2, ExcNotImplemented());
    
    // check the x coordinate of the point
    if (p[0] <= 0.55*L)
    return P1;
    else
    return P2;
}
