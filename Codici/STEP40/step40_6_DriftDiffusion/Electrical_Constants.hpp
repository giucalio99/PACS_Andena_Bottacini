#ifndef ELECTRICAL_CONSTANTS_HPP
#define ELECTRICAL_CONSTANTS_HPP

#include<cmath>
// This header file stores all the constants used in the validation of the pn junction problem

const double eps_0 = 8.854 * 1.e-12;    // [F/m]= [ (C^2 * s^2) / (kg * m^3)] = [ C^2 /(N * m^2)] Permittivity of free space
const double eps_r = 4.;                // [ADIM] Relative permittivity
const double q0 = 1.602 * 1.e-19;       // [C] Elementary charge

const double L = 1.5e-6;                // Length of the domain

const double A = 1.e+22;                // Negative doping value (when we operate with this value we put a minus sign before)
const double D = 1.e+22;                // Positive doping value. They are both constant in time

const double ni = 1.e+10;   //forse 1e+16?              // constant starting ion density, charge density of ions in the ambient

const double N1 = D/2. + std::sqrt(D*D + 4.*ni*ni)/2.; // [m^-3] Electron density on boundary 1
const double P2 = A/2. + std::sqrt(A*A + 4.*ni*ni)/2.; // [m^-3] Holes density on boundary 2   

const double P1 = ni*ni/N1; // [m^-3] Holes density on boundary 1
const double N2 = ni*ni/P2; // [m^-3] Electron density on boundary 2

const double V_TH = 2.6e-2;    // [V] ion temperature in Volts for both 


#endif // ELECTRICAL_CONSTANTS_HPP