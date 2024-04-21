#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include<cmath>    // magari creare un namespace per le costanti ?? e poi usare using namespace bla bla?

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
const double Re = 2.5e-5; // [m] emitter radius
const double X = -Re; // [m] emitter center

const double g = 0.02; // [m] interelectrode distance

// collector
const double collector_length = 0.1; // [m]

// Peek's law (empyrical)
const double eps = 1.; // wire surface roughness correction coefficient
const double Ep = E_ON*delta*eps*(1+0.308/std::sqrt(Re*1.e+2*delta));
const double Ri = Ep/E_ON*Re; // [m] ionization radius
const double Vi = Ve - Ep*Re*std::log(Ep/E_ON); // [V] voltage on ionization region boundary



#endif // CONSTANTS_HPP