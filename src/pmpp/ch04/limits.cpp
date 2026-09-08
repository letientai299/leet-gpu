#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optional>
#include <string>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace {

constexpr double bytes_per_mib = 1024.0 * 1024.0;

struct LinkInfo {
  const char* title;
  const char* url;
};

constexpr std::array links{
    LinkInfo{"CUDA Runtime device management",
             "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html"},
    LinkInfo{"CUDA device properties",
             "https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html"},
    LinkInfo{"CUDA occupancy", "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/"
                               "writing-cuda-kernels.html"},
    LinkInfo{"CUDA compute capabilities",
             "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/"
             "compute-capabilities.html"},
    LinkInfo{"CUDA programming model",
             "https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/"
             "programming-model.html"},
    LinkInfo{"CUDA Runtime types",
             "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html"},
    LinkInfo{"CUDA Driver device management",
             "https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html"},
    LinkInfo{"CUDA binary utilities", "https://docs.nvidia.com/cuda/cuda-binary-utilities/"},
    LinkInfo{"NVCC", "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/"},
    LinkInfo{"Nsight Compute", "https://docs.nvidia.com/nsight-compute/"},
    LinkInfo{"nvidia-smi", "https://docs.nvidia.com/deploy/nvidia-smi/"},
};

#ifdef _WIN32
using LibraryHandle = HMODULE;

LibraryHandle open_driver() {
  return LoadLibraryA("nvcuda.dll");
}

void close_driver(LibraryHandle library) {
  FreeLibrary(library);
}
#else
using LibraryHandle = void*;

LibraryHandle open_driver() {
  return dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
}

void close_driver(LibraryHandle library) {
  dlclose(library);
}
#endif

template <typename Function> Function load_function(LibraryHandle library, const char* name) {
#ifdef _WIN32
  return reinterpret_cast<Function>(GetProcAddress(library, name));
#else
  return reinterpret_cast<Function>(dlsym(library, name));
#endif
}

class DriverApi {
public:
  using Init = CUresult(CUDAAPI*)(unsigned int);
  using GetDevice = CUresult(CUDAAPI*)(CUdevice*, int);
  using GetAttribute = CUresult(CUDAAPI*)(int*, CUdevice_attribute, CUdevice);

  DriverApi()
      : library_(open_driver()), init_(load<Init>("cuInit")),
        get_device_(load<GetDevice>("cuDeviceGet")),
        get_attribute_(load<GetAttribute>("cuDeviceGetAttribute")) {
  }

  DriverApi(const DriverApi&) = delete;
  DriverApi& operator=(const DriverApi&) = delete;

  ~DriverApi() {
    if (library_ != nullptr) {
      close_driver(library_);
    }
  }

  [[nodiscard]] bool available() const {
    return init_ != nullptr && get_device_ != nullptr && get_attribute_ != nullptr;
  }

  [[nodiscard]] CUresult init() const {
    return init_(0);
  }
  [[nodiscard]] CUresult get_device(CUdevice* device, int ordinal) const {
    return get_device_(device, ordinal);
  }
  [[nodiscard]] CUresult get_attribute(int* value, CUdevice_attribute attr, CUdevice device) const {
    return get_attribute_(value, attr, device);
  }

private:
  template <typename Function> Function load(const char* name) const {
    return library_ == nullptr ? nullptr : load_function<Function>(library_, name);
  }

  LibraryHandle library_{};
  Init init_{};
  GetDevice get_device_{};
  GetAttribute get_attribute_{};
};

namespace color {

constexpr auto reset = "\033[0m";
constexpr auto heading = "\033[1;33m";
constexpr auto title = "\033[1;36m";

} // namespace color

bool color_enabled() {
  static const bool enabled = [] {
    if (std::getenv("NO_COLOR") != nullptr) {
      return false;
    }
#ifdef _WIN32
    return _isatty(_fileno(stdout)) != 0;
#else
    return isatty(fileno(stdout)) != 0;
#endif
  }();
  return enabled;
}

const char* paint(const char* code) {
  return color_enabled() ? code : "";
}

void print_section(const char* title) {
  std::printf("\n==> %s%s%s\n", paint(color::heading), title, paint(color::reset));
}

void print_row(const char* title, const std::string& value) {
  std::printf("%s%-38s%s  %s\n", paint(color::title), title, paint(color::reset), value.c_str());
}

void print_links() {
  print_section("Learn");
  for (const auto& link : links) {
    std::printf("%s\n  %s\n", link.title, link.url);
  }
}

bool cuda_ok(cudaError_t error, const char* operation) {
  if (error == cudaSuccess) {
    return true;
  }
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
  return false;
}

std::string format_bool(int value) {
  return value ? "yes" : "no";
}

std::string format_number(int value) {
  return std::to_string(value);
}

std::string format_bytes(std::size_t value) {
  std::array<char, 64> output{};
  std::snprintf(output.data(), output.size(), "%zu bytes (%.2f MiB)", value,
                static_cast<double>(value) / bytes_per_mib);
  return output.data();
}

std::string format_version(int version) {
  if (version == 0) {
    return "not installed";
  }
  std::array<char, 32> output{};
  std::snprintf(output.data(), output.size(), "%d.%d (%d)", version / 1000, version % 1000 / 10,
                version);
  return output.data();
}

std::string format_hex(unsigned int value) {
  std::array<char, 16> output{};
  std::snprintf(output.data(), output.size(), "0x%x", value);
  return output.data();
}

std::string format_clock(int value) {
  return std::to_string(value) + " kHz";
}

std::string format_mask(int value) {
  return format_hex(static_cast<unsigned>(value));
}

std::string format_int_bytes(int value) {
  return value < 0 ? "unavailable" : format_bytes(static_cast<std::size_t>(value));
}

std::string format_count(std::size_t value) {
  return std::to_string(value);
}

std::string format_compute_mode(int mode) {
  switch (static_cast<cudaComputeMode>(mode)) {
  case cudaComputeModeDefault:
    return "default";
  case cudaComputeModeExclusive:
    return "exclusive thread";
  case cudaComputeModeProhibited:
    return "prohibited";
  case cudaComputeModeExclusiveProcess:
    return "exclusive process";
  }
  return "unknown";
}

template <std::size_t Size> std::string format_dims(const int (&values)[Size]) {
  std::array<char, 64> output{};
  if constexpr (Size == 2) {
    std::snprintf(output.data(), output.size(), "%d x %d", values[0], values[1]);
  } else {
    std::snprintf(output.data(), output.size(), "%d x %d x %d", values[0], values[1], values[2]);
  }
  return output.data();
}

std::string format_uuid(const cudaUUID_t& uuid) {
  std::array<char, 37> output{};
  std::snprintf(
      output.data(), output.size(),
      "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
      static_cast<unsigned char>(uuid.bytes[0]), static_cast<unsigned char>(uuid.bytes[1]),
      static_cast<unsigned char>(uuid.bytes[2]), static_cast<unsigned char>(uuid.bytes[3]),
      static_cast<unsigned char>(uuid.bytes[4]), static_cast<unsigned char>(uuid.bytes[5]),
      static_cast<unsigned char>(uuid.bytes[6]), static_cast<unsigned char>(uuid.bytes[7]),
      static_cast<unsigned char>(uuid.bytes[8]), static_cast<unsigned char>(uuid.bytes[9]),
      static_cast<unsigned char>(uuid.bytes[10]), static_cast<unsigned char>(uuid.bytes[11]),
      static_cast<unsigned char>(uuid.bytes[12]), static_cast<unsigned char>(uuid.bytes[13]),
      static_cast<unsigned char>(uuid.bytes[14]), static_cast<unsigned char>(uuid.bytes[15]));
  return output.data();
}

std::optional<int> query_attr(cudaDeviceAttr attr, int device) {
  int value = 0;
  if (cudaDeviceGetAttribute(&value, attr, device) != cudaSuccess) {
    return std::nullopt;
  }
  return value;
}

const DriverApi& driver() {
  static const DriverApi api;
  return api;
}

std::optional<int> query_driver_attr(CUdevice_attribute attr, CUdevice device) {
  int value = 0;
  if (driver().get_attribute(&value, attr, device) != CUDA_SUCCESS) {
    return std::nullopt;
  }
  return value;
}

using IntFormatter = std::string (*)(int);

struct DriverAttribute {
  const char* title;
  CUdevice_attribute attr;
  IntFormatter formatter{format_bool};
};

constexpr std::array driver_attributes{
    DriverAttribute{"Virtual memory management",
                    CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED},
    DriverAttribute{"Generic memory compression",
                    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED},
    DriverAttribute{"GPUDirect RDMA with VMM",
                    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED},
    DriverAttribute{"64-bit stream memory ops", CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS},
    DriverAttribute{"Stream wait-value NOR", CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR},
    DriverAttribute{"Memory sync domains", CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT,
                    format_number},
    DriverAttribute{"Tensor Map access", CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED},
    DriverAttribute{"Fabric memory handles", CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED},
    DriverAttribute{"Multicast operations", CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED},
    DriverAttribute{"Memory decompression algorithms",
                    CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK, format_mask},
    DriverAttribute{"Maximum decompression length",
                    CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH, format_int_bytes},
    DriverAttribute{"Host NUMA virtual memory",
                    CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED},
    DriverAttribute{"Host NUMA memory pools", CU_DEVICE_ATTRIBUTE_HOST_NUMA_MEMORY_POOLS_SUPPORTED},
    DriverAttribute{"Host memory pools", CU_DEVICE_ATTRIBUTE_HOST_MEMORY_POOLS_SUPPORTED},
    DriverAttribute{"Host virtual memory",
                    CU_DEVICE_ATTRIBUTE_HOST_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED},
    DriverAttribute{"Host allocation DMA-BUF", CU_DEVICE_ATTRIBUTE_HOST_ALLOC_DMA_BUF_SUPPORTED},
    DriverAttribute{"DMA-BUF mmap", CU_DEVICE_ATTRIBUTE_DMA_BUF_MMAP_SUPPORTED},
    DriverAttribute{"Partial host native atomics",
                    CU_DEVICE_ATTRIBUTE_ONLY_PARTIAL_HOST_NATIVE_ATOMIC_SUPPORTED},
    DriverAttribute{"Atomic reductions", CU_DEVICE_ATTRIBUTE_ATOMIC_REDUCTION_SUPPORTED},
    DriverAttribute{"Logical endpoint unicast",
                    CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_SUPPORTED},
    DriverAttribute{"Logical endpoint multicast",
                    CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_MULTICAST_SUPPORTED},
    DriverAttribute{"Logical endpoint counted ops",
                    CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_COUNTED_OPS_SUPPORTED},
    DriverAttribute{"Owner-device endpoint access",
                    CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_ACCESS_ON_OWNER_DEVICE_SUPPORTED},
};

template <typename Attribute, typename Device, typename Query>
void print_query(
    const char* title, Attribute attr, Device device, Query query, IntFormatter formatter) {
  const auto value = query(attr, device);
  print_row(title, value ? formatter(*value) : "unavailable");
}

void print_attr(const char* title,
                cudaDeviceAttr attr,
                int device,
                IntFormatter formatter = format_number) {
  print_query(title, attr, device, query_attr, formatter);
}

void print_driver_attr(const char* title,
                       CUdevice_attribute attr,
                       CUdevice device,
                       IntFormatter formatter = format_number) {
  print_query(title, attr, device, query_driver_attr, formatter);
}

using SizeFormatter = std::string (*)(std::size_t);

void print_limit(const char* title, cudaLimit limit, SizeFormatter formatter = format_bytes) {
  std::size_t value = 0;
  if (cudaDeviceGetLimit(&value, limit) != cudaSuccess) {
    print_row(title, "unavailable");
    return;
  }
  print_row(title, formatter(value));
}

void print_identity(int device, const cudaDeviceProp& prop) {
  std::array<char, 32> pci_bus{};
  if (!cuda_ok(cudaDeviceGetPCIBusId(pci_bus.data(), static_cast<int>(pci_bus.size()), device),
               "cudaDeviceGetPCIBusId")) {
    pci_bus[0] = '\0';
  }

  const std::string section = "Device " + std::to_string(device);
  print_section(section.c_str());
  print_row("Name", prop.name);
  print_row("Compute capability", std::to_string(prop.major) + "." + std::to_string(prop.minor) +
                                      " (sm_" + std::to_string(prop.major) +
                                      std::to_string(prop.minor) + ")");
  print_row("PCI bus ID", pci_bus.data());
  print_row("PCI device/vendor ID", format_hex(prop.gpuPciDeviceID));
  print_row("PCI subsystem/vendor ID", format_hex(prop.gpuPciSubsystemID));
  print_row("UUID", format_uuid(prop.uuid));
  print_row("Streaming multiprocessors", std::to_string(prop.multiProcessorCount));
  print_row("Integrated GPU", format_bool(prop.integrated));
}

void print_occupancy(int device, const cudaDeviceProp& prop) {
  print_section("Occupancy limits");
  print_row("Warp size", std::to_string(prop.warpSize));
  print_row("Max resident threads per SM", std::to_string(prop.maxThreadsPerMultiProcessor));
  const int warps = prop.warpSize == 0 ? 0 : prop.maxThreadsPerMultiProcessor / prop.warpSize;
  print_row("Max resident warps per SM", std::to_string(warps) + " (derived)");
  print_row("Max resident blocks per SM", std::to_string(prop.maxBlocksPerMultiProcessor));
  print_row("Max threads per block", std::to_string(prop.maxThreadsPerBlock));
  print_row("Max block dimensions", format_dims(prop.maxThreadsDim));
  print_row("Max grid dimensions", format_dims(prop.maxGridSize));
  print_row("Registers per SM", std::to_string(prop.regsPerMultiprocessor));
  print_row("Registers per block", std::to_string(prop.regsPerBlock));
  print_row("Shared memory per SM", format_bytes(prop.sharedMemPerMultiprocessor));
  print_row("Shared memory per block", format_bytes(prop.sharedMemPerBlock));
  print_row("Opt-in shared memory per block", format_bytes(prop.sharedMemPerBlockOptin));
  print_row("Reserved shared memory per block", format_bytes(prop.reservedSharedMemPerBlock));
  print_row("Kernel occupancy", "requires kernel launch config");
  print_attr("SM clock limit", cudaDevAttrClockRate, device, format_clock);
}

void print_memory(int device, const cudaDeviceProp& prop) {
  print_section("Memory limits");
  print_row("Global memory", format_bytes(prop.totalGlobalMem));
  print_row("Constant memory", format_bytes(prop.totalConstMem));
  print_row("L2 cache", format_bytes(static_cast<std::size_t>(prop.l2CacheSize)));
  print_row("Persistent L2 cache", format_bytes(prop.persistingL2CacheMaxSize));
  print_row("Maximum memory pitch", format_bytes(prop.memPitch));
  print_row("Memory bus width", std::to_string(prop.memoryBusWidth) + " bits");
  print_attr("Memory clock limit", cudaDevAttrMemoryClockRate, device, format_clock);
  print_row("Access policy window", format_bytes(prop.accessPolicyMaxWindowSize));
  print_row("Global L1 cache", format_bool(prop.globalL1CacheSupported));
  print_row("Local L1 cache", format_bool(prop.localL1CacheSupported));
}

void print_textures(const cudaDeviceProp& prop) {
  print_section("Texture limits");
  print_row("Max texture 1D", std::to_string(prop.maxTexture1D));
  print_row("Max texture 1D mipmapped", std::to_string(prop.maxTexture1DMipmap));
  print_row("Max texture 2D", format_dims(prop.maxTexture2D));
  print_row("Max texture 2D mipmapped", format_dims(prop.maxTexture2DMipmap));
  print_row("Max texture 2D linear", format_dims(prop.maxTexture2DLinear));
  print_row("Max texture 2D gather", format_dims(prop.maxTexture2DGather));
  print_row("Max texture 3D", format_dims(prop.maxTexture3D));
  print_row("Max texture 3D alternate", format_dims(prop.maxTexture3DAlt));
  print_row("Max texture cubemap", std::to_string(prop.maxTextureCubemap));
  print_row("Max texture 1D layered", format_dims(prop.maxTexture1DLayered));
  print_row("Max texture 2D layered", format_dims(prop.maxTexture2DLayered));
  print_row("Max texture cubemap layered", format_dims(prop.maxTextureCubemapLayered));
  print_row("Texture alignment", format_bytes(prop.textureAlignment));
  print_row("Texture pitch alignment", format_bytes(prop.texturePitchAlignment));
}

void print_surfaces(const cudaDeviceProp& prop) {
  print_section("Surface limits");
  print_row("Max surface 1D", std::to_string(prop.maxSurface1D));
  print_row("Max surface 2D", format_dims(prop.maxSurface2D));
  print_row("Max surface 3D", format_dims(prop.maxSurface3D));
  print_row("Max surface 1D layered", format_dims(prop.maxSurface1DLayered));
  print_row("Max surface 2D layered", format_dims(prop.maxSurface2DLayered));
  print_row("Max surface cubemap", std::to_string(prop.maxSurfaceCubemap));
  print_row("Max surface cubemap layered", format_dims(prop.maxSurfaceCubemapLayered));
  print_row("Surface alignment", format_bytes(prop.surfaceAlignment));
}

void print_capabilities(int device, const cudaDeviceProp& prop) {
  print_section("Capabilities");
  print_attr("Compute mode", cudaDevAttrComputeMode, device, format_compute_mode);
  print_attr("Kernel execution timeout", cudaDevAttrKernelExecTimeout, device, format_bool);
  print_attr("Concurrent copy and execution", cudaDevAttrGpuOverlap, device, format_bool);
  print_attr("FP32 to FP64 throughput ratio", cudaDevAttrSingleToDoublePrecisionPerfRatio, device);
  print_row("Concurrent kernels", format_bool(prop.concurrentKernels));
  print_row("Asynchronous engines", std::to_string(prop.asyncEngineCount));
  print_row("Unified addressing", format_bool(prop.unifiedAddressing));
  print_row("Managed memory", format_bool(prop.managedMemory));
  print_row("Concurrent managed access", format_bool(prop.concurrentManagedAccess));
  print_row("Pageable memory access", format_bool(prop.pageableMemoryAccess));
  print_row("Pageable access uses host tables",
            format_bool(prop.pageableMemoryAccessUsesHostPageTables));
  print_row("Direct managed host access", format_bool(prop.directManagedMemAccessFromHost));
  print_row("Map host memory", format_bool(prop.canMapHostMemory));
  print_row("Registered host pointer identity",
            format_bool(prop.canUseHostPointerForRegisteredMem));
  print_row("Host memory registration", format_bool(prop.hostRegisterSupported));
  print_row("Read-only host registration", format_bool(prop.hostRegisterReadOnlySupported));
  print_row("Host native atomics", format_bool(prop.hostNativeAtomicSupported));
  print_row("Compute preemption", format_bool(prop.computePreemptionSupported));
  print_row("Stream priorities", format_bool(prop.streamPrioritiesSupported));
  print_row("Cooperative launch", format_bool(prop.cooperativeLaunch));
  print_row("Memory pools", format_bool(prop.memoryPoolsSupported));
  print_row("Sparse CUDA arrays", format_bool(prop.sparseCudaArraySupported));
  print_row("Deferred CUDA array mapping", format_bool(prop.deferredMappingCudaArraySupported));
  print_row("Cluster launch", format_bool(prop.clusterLaunch));
  print_row("GPUDirect RDMA", format_bool(prop.gpuDirectRDMASupported));
  print_row("GPUDirect RDMA flush options", format_hex(prop.gpuDirectRDMAFlushWritesOptions));
  print_row("GPUDirect RDMA write ordering", std::to_string(prop.gpuDirectRDMAWritesOrdering));
  print_row("Memory-pool handle types", format_hex(prop.memoryPoolSupportedHandleTypes));
  print_row("Timeline semaphore interop", format_bool(prop.timelineSemaphoreInteropSupported));
  print_row("IPC events", format_bool(prop.ipcEventSupported));
  print_row("Unified function pointers", format_bool(prop.unifiedFunctionPointers));
  print_row("MPS enabled", format_bool(prop.mpsEnabled));
  print_row("TCC driver", format_bool(prop.tccDriver));
  print_row("Multi-GPU board", format_bool(prop.isMultiGpuBoard));
  print_row("Multi-GPU board group", std::to_string(prop.multiGpuBoardGroupID));
  print_row("Device NUMA ID", std::to_string(prop.deviceNumaId));
  print_row("Device NUMA configuration", std::to_string(prop.deviceNumaConfig));
  print_row("Nearest host NUMA ID", std::to_string(prop.hostNumaId));
  print_row("Host NUMA multinode IPC", format_bool(prop.hostNumaMultinodeIpcSupported));
  print_row("ECC enabled", format_bool(prop.ECCEnabled));
}

void print_driver_capabilities(int device) {
  const auto& api = driver();
  CUdevice driver_device{};
  if (!api.available() || api.init() != CUDA_SUCCESS ||
      api.get_device(&driver_device, device) != CUDA_SUCCESS) {
    return;
  }

  print_section("Driver capabilities");
  for (const auto& attribute : driver_attributes) {
    print_driver_attr(attribute.title, attribute.attr, driver_device, attribute.formatter);
  }
}

void print_runtime_limits(int device) {
  print_section("Runtime-configured limits");
  if (!cuda_ok(cudaSetDevice(device), "cudaSetDevice")) {
    return;
  }
  print_limit("Thread stack size", cudaLimitStackSize);
  print_limit("Device printf FIFO", cudaLimitPrintfFifoSize);
  print_limit("Device malloc heap", cudaLimitMallocHeapSize);
  print_limit("Pending device launches", cudaLimitDevRuntimePendingLaunchCount, format_count);
  print_limit("Maximum L2 fetch granularity", cudaLimitMaxL2FetchGranularity);
  print_limit("Persisting L2 set-aside", cudaLimitPersistingL2CacheSize);
}

void print_peer_access(int device_count) {
  if (device_count < 2) {
    return;
  }

  print_section("Peer access");
  for (int source = 0; source < device_count; ++source) {
    for (int target = 0; target < device_count; ++target) {
      if (source == target) {
        continue;
      }

      int access = 0;
      int rank = 0;
      int atomics = 0;
      int arrays = 0;
      const bool available =
          cudaDeviceCanAccessPeer(&access, source, target) == cudaSuccess &&
          cudaDeviceGetP2PAttribute(&rank, cudaDevP2PAttrPerformanceRank, source, target) ==
              cudaSuccess &&
          cudaDeviceGetP2PAttribute(&atomics, cudaDevP2PAttrNativeAtomicSupported, source,
                                    target) == cudaSuccess &&
          cudaDeviceGetP2PAttribute(&arrays, cudaDevP2PAttrCudaArrayAccessSupported, source,
                                    target) == cudaSuccess;
      const std::string title =
          "Device " + std::to_string(source) + " -> " + std::to_string(target);
      if (!available) {
        print_row(title.c_str(), "unavailable");
        continue;
      }
      const std::string value = format_bool(access) + ", rank " + std::to_string(rank) +
                                ", atomics " + format_bool(atomics) + ", arrays " +
                                format_bool(arrays);
      print_row(title.c_str(), value);
    }
  }
}

void print_device(int device) {
  cudaDeviceProp prop{};
  if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties")) {
    return;
  }
  print_identity(device, prop);
  print_occupancy(device, prop);
  print_memory(device, prop);
  print_textures(prop);
  print_surfaces(prop);
  print_capabilities(device, prop);
  print_driver_capabilities(device);
  print_runtime_limits(device);
}

void print_commands() {
  print_section("Cross-check commands");
  print_row("nvidia-smi -L", "GPU indices, names, and UUIDs");
  print_row("nvidia-smi -q", "Driver, memory, clocks, health");
  print_row("nvidia-smi --query-gpu=...", "Identity and memory fields");
  print_row("nvidia-smi topo -m", "GPU, NIC, CPU affinity topology");
  print_row("nvcc --version", "CUDA toolkit compiler version");
  print_row("deviceQuery", "Runtime properties, when installed");
  print_row("cuobjdump -res-usage <binary>", "Per-kernel registers and memory");
  print_row("ncu --section Occupancy <app>", "Kernel theoretical occupancy");
}

void print_help() {
  print_commands();
  print_links();
}

} // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);

  int driver_version = 0;
  if (cuda_ok(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion")) {
    print_row("CUDA driver API version", format_version(driver_version));
  }

  int runtime_version = 0;
  if (cuda_ok(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion")) {
    print_row("CUDA runtime version", format_version(runtime_version));
  }

  int device_count = 0;
  if (!cuda_ok(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount")) {
    print_help();
    return 1;
  }
  print_row("Visible CUDA devices", std::to_string(device_count));
  for (int device = 0; device < device_count; ++device) {
    print_device(device);
  }
  print_peer_access(device_count);
  print_help();
  return device_count == 0 ? 1 : 0;
}
