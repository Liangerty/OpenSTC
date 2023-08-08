#include "MPIIO.hpp"

void cfd::write_str(const char *str, MPI_File &file, MPI_Offset &offset) {
  int value = 0;
  while (*str != '\0') {
    value = static_cast<int>(*str);
    MPI_File_write_at(file, offset, &value, 1, MPI_INT32_T, MPI_STATUS_IGNORE);
    offset+=4;
    ++str;
  }
  constexpr char null_char = '\0';
  value = static_cast<int>(null_char);
  MPI_File_write_at(file, offset, &value, 1, MPI_INT32_T, MPI_STATUS_IGNORE);
  offset+=4;
}

void cfd::write_str_without_null(const char *str, MPI_File &file, MPI_Offset &offset) {
//  int value = 0;
  while (*str != '\0') {
//    value = static_cast<int>(*str);
    MPI_File_write_at(file, offset, str, 1, MPI_CHAR, MPI_STATUS_IGNORE);
    ++offset;
    ++str;
  }
}
