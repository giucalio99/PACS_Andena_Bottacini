#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP   

// ATTENZIONE
// vedere se coerente con ELECTRICAL CONSTANTS di validation_pn; usare solo una delle due
// CONSTANTS file, quello più completo nel file finale                  

// Physical Constants
const double eps_0 = 8.854 * 1.e-12; //[F/m]= [C^2 s^2 / kg / m^3]
const double eps_r = 1.0006;
const double q0 = 1.602 * 1.e-19; // [C]
const double kB = 1.381 * 1.e-23 ; //[J/K]

const bool stratosphere = false; // if false, use atmospheric 0 km conditions

const bool Dirichlet = false; // if true, use homogeneous dirichlet conditions at collector electrode
const bool Neumann = false; // if true, use homogeneous neumann conditions at collector electrode
// if both Dirichlet and Neumann are false, a Robin condition is used

const double q_over_eps_0 = q0 / eps_0; // [m^3 kg C^-1 s^-2]
const double mu = 1.83e-4; // [m^2/s/V]
const double T = stratosphere ? 217. : 303.; // [K] fluid temperature
const double V_E = kB * T / q0; // [V] ion temperature
const double D = mu * V_E;
const double rho = stratosphere ? 0.089 : 1.225; // kg m^-3
const double Mm = 29.e-3; // kg m^-3,average air molar mass
const double Avo = 6.022e+23; // Avogadro's number
//const double n_air = rho / Mm * Avo; // m^-3

// Input Parameters
const double Om_0 = 0.; // source/sink ion term [m^-3 s^-1]
const double E_ON = 3.31e+6; // onset field threshold [V/m]
const double E_ref = 1.e+7; // [V/m] maximum field value
const double N_ref = 1.e+14; // [m^-3] maximum density value
const double N_0 = stratosphere ? 2.2e-3 : 0.5e-3; // [m^-3] ambient ion density
const double p_out = stratosphere ? 5474 : 101300;
const double p_over_rho = 0.;//p_out / rho;

// Geometry Data (always make it consistent with gmsh file)

// emitter
const double Vmax = 2.e+4; // [V] emitter voltage
const double R = 2.5e-4; // [m] wire radius
const double X = -R; // [m] emitter center

const double g = 0.02; // [m] interelectrode distance
const double mesh_height = 0.02;; // [m]

// collector
const double collector_height = 0.002; // [m]
const double collector_length = 0.01; // [m]
const double XC = g + collector_length/2.; // [m]



#endif // CONSTANTS_HPP