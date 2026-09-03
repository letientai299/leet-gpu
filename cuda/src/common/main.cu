#include "cuda_check.hpp"
#include "kernel.hpp"
#include "log.hpp"

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
  init_log();
  if (argc == 2 && strcmp(argv[1], "--help") == 0) {
    print_help(argv[0]);
    return 0;
  }

  const char *name = select_kernel(argc, argv);
  const Kernel *kernel = find_kernel(name);
  if (!kernel) {
    if (name) {
      HOST_LOG("Unknown kernel: %s", name);
    }
    print_help(argv[0]);
    return 2;
  }

  int device_id = 0;
  cudaDeviceProp device{};
  if (!CUDA_CHECK(cudaGetDevice(&device_id)) ||
      !CUDA_CHECK(cudaGetDeviceProperties(&device, device_id))) {
    return 1;
  }
  set_gpu_arch(device.major, device.minor);
  if (!supports(*kernel, device)) {
    HOST_LOG("%s requires sm_%d%d%s; device is sm_%d%d", kernel->name,
             kernel->major, kernel->minor, kernel->exact ? " exactly" : "+",
             device.major, device.minor);
    return 2;
  }

  HOST_LOG("%s on %s (sm_%d%d)", name, device.name, device.major, device.minor);
  return kernel->run() ? 0 : 1;
}
