#pragma once

struct Kernel {
  const char *name;
  const char *description;
  int major;
  int minor;
  bool exact;
  bool (*run)();
};

const Kernel *find_kernel(const char *name);
void print_help(const char *program);
