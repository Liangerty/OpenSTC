#include "Initialize.cuh"
#include <cstdio>

void cfd::read_one_useless_variable(FILE *fp, integer mx, integer my, integer mz, integer data_format) {
  if (data_format == 1) {
    // float
    float v{0.0f};
    for (int k = 0; k < mz; ++k) {
      for (int j = 0; j < my; ++j) {
        for (int i = 0; i < mx; ++i) {
          fread(&v, 4, 1, fp);
        }
      }
    }
  } else {
    // double
    double v{0.0};
    for (int k = 0; k < mz; ++k) {
      for (int j = 0; j < my; ++j) {
        for (int i = 0; i < mx; ++i) {
          fread(&v, 8, 1, fp);
        }
      }
    }
  }
}
