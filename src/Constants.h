#pragma once

#include <limits>
/**
 * \brief The physical constants used in our simulation
 *
 * \ref [NIST Reference on Constants, Units and Uncertanity](https://physics.nist.gov/cuu/Reference/contents.html)
 *
 **/
namespace cfd {
// Avogadro's Number [kmole^{-1}]
constexpr double avogadro = 6.02214076e26;
// Elementary charge  [C]
constexpr double electron_charge = 1.602176634e-19;
// Electron Mass  [kg]
constexpr double electron_mass = 9.1093837015e-31;
// Universal gas constant [J/(kmole*K)]
constexpr double R_u = 8314.462618;
// Universal gas constant [cal/(mole*K)], fetched from [Gas constant](https://en.wikipedia.org/wiki/Gas_constant)
constexpr double R_c = 1.98720425864083;
// Specific heat ratio for air in perfect gas
constexpr double gamma_air = 1.4;
// Atmospheric pressure
constexpr double p_atm = 101325;
// Air molecular weight.
// Ref: Engineering ToolBox, (2003). Air - Thermophysical Properties. [online] Available at: https://www.engineeringtoolbox.com/air-properties-d_156.html [Accessed Day Mo. Year].
constexpr double mw_air = 28.9647;
}
