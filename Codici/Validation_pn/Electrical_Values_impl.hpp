// This .hpp file contains the definitions of the override methods "value" of the four different template functions declared
// in "Electrical_Values.hpp" file.
// These template classes are used to initialize the value of the problem

using namespace dealii; // in order to avoid dealii:: every time

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
    return E;
    else
    return -A;

}

// --------------------------------------- POTENTIAL VALUES ---------------------------------------------------------------------------------------------------

// this method return the value of the potential, if we are in the first 45% of the domain (x-axis) V_E*log(N1/N0) otherwise V_E*log(N2/N0).

template <int dim>
double PotentialValues<dim>::value(const Point<dim> & p, const unsigned int component) const{
    
    // these 3 lines are identical to the first template, same reasoning
    (void)component;
    AssertIndexRange(component, 1);
    Assert(dim == 2, ExcNotImplemented());

    if (p[0] <= 0.45*L) // check the x component of the point p
    return V_E*std::log(N1/N_0);
    else
    return V_E*std::log(N2/N_0);

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

// This method return the value of the ion in the domain: we initialize the first 55% of the domain to P1 the rest P2

template <int dim>
double IonInitialValues<dim>::value(const Point<dim> & p, const unsigned int component) const {
    
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


// NOTA DI SERVIZIO: NON C'E' RIUTILIZZO DEL CODICE, NEL SENSO, QUESTE FUNZIONI FANNO PRATICAMANTE LA STESSA COSA
// SI POTREBBE CREARE UN AGGREGATO DI PUNTI { DOPING , POTENTIAL , ELECTRON , ION} PASSARLO E VERIFICARE LE VARIE RICHIESTE, INVECE
// DI USARE 4 TEMPLATE DIVERSI. NON SO SE SIA POSSIBILE QUA SEMBREREBBE CHE AD OGNI PUNTO CHE PASSO ASSOCIO UN SOLO VALORE 