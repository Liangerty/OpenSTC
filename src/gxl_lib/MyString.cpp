#include "MyString.h"
#include <fmt/core.h>

void gxl::trim_left(std::string& string) {
  const auto start =string.find_first_not_of(whiteSpace);
  if (start!=std::string::npos)
    string =string.substr(start);
  else
    string="";
}

std::istream& gxl::getline(std::ifstream& file, std::string& input, Case u_l) {
  std::getline(file,input);
  trim_left(input);
  if (u_l==Case::upper) {
    for(auto&ch:input)
      ch=std::toupper(ch);
  }
  else if (u_l==Case::lower) {
    for(auto&ch:input)
      ch=std::tolower(ch);
  }
  return file;
}

void gxl::to_stringstream(const std::string& input, std::istringstream& line) {
  line.clear();
  line.str(input);
}

std::istream& gxl::getline_to_stream(std::ifstream& file, std::string& input,
                                     std::istringstream& line, Case u_l) {
  std::getline(file,input);
  trim_left(input);
  if (u_l==Case::upper) {
    for(auto&ch:input)
      ch=std::toupper(ch);
  }
  else if (u_l==Case::lower) {
    for(auto&ch:input)
      ch=std::tolower(ch);
  }
  line.clear();
  line.str(input);
  return file;
}

void gxl::read_until(std::ifstream& file, std::string& input,
                     std::string&& to_find, Case u_l) {
  while (getline(file,input,u_l)) {
    if (input.starts_with(to_find))
      return;
  }
  input = "end"; // 若没找到，就赋值为"end"代表结束了
  fmt::print("{} is not found in file.\n", to_find);
}

std::string gxl::to_upper(std::string& str) {
  for (auto& ch : str)
    ch = std::toupper(ch);
  return str;
}

std::string gxl::read_str(FILE* file) {
  int value = 0;
  std::string str;
  while (true) {
    fread(&value, sizeof(int), 1, file);
    const char ch = static_cast<char>(value);
    if (ch == '\0')
      break;
    str += ch;
  }
  return str;
}

std::string gxl::to_upper(const std::string &str) {
  std::string str_upper{str};
  for (auto& ch:str_upper)
    ch=std::toupper(ch);
  return str_upper;
}

void gxl::write_str(const char *str, FILE *file) {
  int value = 0;
  while (*str != '\0') {
    value = static_cast<int>(*str);
    fwrite(&value, sizeof(int), 1, file);
    ++str;
  }
  constexpr char null_char = '\0';
  value                    = static_cast<int>(null_char);
  fwrite(&value, sizeof(int), 1, file);
}
