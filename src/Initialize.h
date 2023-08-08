#pragma once

#include "Parameter.h"
#include "Field.h"

#if MULTISPECIES == 1

#include "ChemData.h"

#endif

namespace cfd {
class Mesh;

void initialize_basic_variables(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field
#if MULTISPECIES == 1
    , ChemData &chem_data
#endif
);

void initialize_from_start(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field
#if MULTISPECIES == 1
    , ChemData &chem_data
#endif
);
}