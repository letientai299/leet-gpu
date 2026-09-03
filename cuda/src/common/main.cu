#include "cuda_check.hpp"
#include "kernel.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

const char *select_kernel(int argc, char **argv) {
  if (argc == 3 && strcmp(argv[1], "--kernel") == 0) {
    return argv[2];
  }
  if (argc == 1) {
    if (const char *name = std::getenv("CUDA_KERNEL")) {
      return name;
    }
  }
  return nullptr;
}

bool supports(const Kernel &kernel, const cudaDeviceProp &device) {
  if (kernel.exact) {
    return device.major == kernel.major && device.minor == kernel.minor;
  }
  return device.major > kernel.major ||
         (device.major == kernel.major && device.minor >= kernel.minor);
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--help") == 0) {
    print_help(argv[0]);
    return 0;
  }

  const char *name = select_kernel(argc, argv);
  const Kernel *kernel = find_kernel(name);
  if (!kernel) {
    if (name) {
      fprintf(stderr, "Unknown kernel: %s\n\n", name);
    }
    print_help(argv[0]);
    return 2;
  }

  int device_id = 0;
  cudaDeviceProp device{};
  if (!cuda_check(cudaGetDevice(&device_id)) ||
      !cuda_check(cudaGetDeviceProperties(&device, device_id))) {
    return 1;
  }
  if (!supports(*kernel, device)) {
    fprintf(stderr, "%s requires sm_%d%d%s; device is sm_%d%d\n", kernel->name,
            kernel->major, kernel->minor, kernel->exact ? " exactly" : "+",
            device.major, device.minor);
    return 2;
  }

  printf("[Host] %s on %s (sm_%d%d)\n", name, device.name, device.major,
         device.minor);
  return kernel->run() ? 0 : 1;
}
