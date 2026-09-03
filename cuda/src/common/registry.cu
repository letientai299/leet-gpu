#include "kernel.hpp"
#include "log.hpp"

#include <array>
#include <cstring>

bool run_hello();
bool run_ampere_async_copy();
bool run_hopper_cluster();
bool run_blackwell_fp4();

namespace {

constexpr std::array<Kernel, 4> kernels{
    Kernel{"hello", "Basic CUDA launch", 8, 6, false, run_hello},
    Kernel{"ampere-async-copy", "Ampere asynchronous shared-memory copy", 8, 6,
           true, run_ampere_async_copy},
    Kernel{"hopper-cluster", "Hopper thread-block cluster", 9, 0, true,
           run_hopper_cluster},
    Kernel{"blackwell-fp4", "Blackwell FP4 conversion", 10, 0, true,
           run_blackwell_fp4},
};

} // namespace

const Kernel *find_kernel(const char *name) {
  if (!name) {
    return nullptr;
  }
  for (const auto &kernel : kernels) {
    if (strcmp(kernel.name, name) == 0) {
      return &kernel;
    }
  }
  return nullptr;
}

void print_help(const char *program) {
  HOST_LOG("Usage: %s --kernel <name>", program);
  HOST_LOG("Kernels:");
  for (const auto &kernel : kernels) {
    HOST_LOG("  %-20s %s", kernel.name, kernel.description);
  }
}
