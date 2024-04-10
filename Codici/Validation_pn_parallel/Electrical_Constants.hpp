#ifndef ELECTRICAL_CONSTANTS_HPP
#define ELECTRICAL_CONSTANTS_HPP

// This header file stores all the constants used in the validation of the pn junction problem

const double eps_0 = 8.854 * 1.e-12;    // [F/m]= [ (C^2 * s^2) / (kg * m^3)] = [ C^2 /(N * m^2)] Permittivity of free space
const double eps_r = 4.;                // [ADIM] Relative permittivity
const double q0 = 1.602 * 1.e-19;       // [C] Elementary charge
const double kB = 1.381 * 1.e-23 ;      // [J/K] Boltzmann constant

const double q_over_eps_0 = q0 / eps_0; // [m^3 kg C^-1 s^-2]
const double L = 1.5e-6;                // Length of the domain
const double A = 1.e+22;                // Negative doping value (when we operate with this value we put a minus sign before)
const double E = 1.e+22;                // Positive doping value ( nella tesi è chiamato D (?))

const double N_0 = 1.e+16;              // constant starting ion density, charge density of ions in the ambient

const double N1 = E/2. + std::sqrt(E*E+4.*N_0*N_0)/2.; // [m^-3] Electron density on boundary 1
const double P2 = A/2. + std::sqrt(A*A+4.*N_0*N_0)/2.; // [m^-3] Electron density on boundary 1   (ELECTRON ??)
const double P1 = N_0*N_0/N1; // [m^-3] Electron density on boundary 2
const double N2 = N_0*N_0/P2; // [m^-3] Electron density on boundary 2

const double mup = 1.e-1;     // [(m^2)/(s*V)] Mobility of positive charges 
const double mun = 3.e-2;     // [(m^2)/(s*V)] Mobility of holes
const double V_E = 2.6e-2;    // [V] ion temperature in Volts for both 

const double Dp = mup * V_E;  // Diffusion coefficient for holes (by Einstein relation)
const double Dn = mun * V_E;  // Diffusion coefficinet for electrons (by Einstein relation)

#endif // ELECTRICAL_CONSTANTS_HPP