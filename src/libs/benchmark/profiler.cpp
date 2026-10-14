#include "benchmark/profiler.hpp"

#include "checks.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cupti_profiler_host.h>
#include <cupti_profiler_target.h>
#include <cupti_range_profiler.h>
#include <cupti_target.h>
#include <map>
#include <memory>
#include <vector>

namespace lg::benchmark {
namespace {

// Nsight Compute's Achieved Occupancy, in percent.
// https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#occupancy-metrics
constexpr const char* kMetric = "sm__warps_active.avg.pct_of_peak_sustained_active";
// The metric needs one pass; the cap stops a runaway replay loop.
constexpr int kMaxPasses = 8;

using Image = std::vector<std::uint8_t>;
using Metrics = std::array<const char*, 1>;

// Range profiler sample: extras/CUPTI/samples/range_profiling.
// https://docs.nvidia.com/cupti/main/main.html#cupti-range-profiling-api
class Host {
public:
  Host() = default;
  ~Host() {
    if (object_ != nullptr) {
      CUpti_Profiler_Host_Deinitialize_Params params{
        CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
      params.pHostObject = object_;
      check_status(cuptiProfilerHostDeinitialize(&params));
    }
  }
  Host(const Host&) = delete;
  Host& operator=(const Host&) = delete;

  bool init(int device, CUcontext context) {
    CUpti_Device_GetChipName_Params chip{CUpti_Device_GetChipName_Params_STRUCT_SIZE};
    chip.deviceIndex = static_cast<std::size_t>(device);
    CUpti_Profiler_GetCounterAvailability_Params availability{
      CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE};
    availability.ctx = context;
    if (!check_status(cuptiDeviceGetChipName(&chip)) ||
        !check_status(cuptiProfilerGetCounterAvailability(&availability))) {
      return false;
    }
    Image counters(availability.counterAvailabilityImageSize);
    availability.pCounterAvailabilityImage = counters.data();
    if (!check_status(cuptiProfilerGetCounterAvailability(&availability))) {
      return false;
    }

    CUpti_Profiler_Host_Initialize_Params params{CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE};
    params.profilerType = CUPTI_PROFILER_TYPE_RANGE_PROFILER;
    params.pChipName = chip.pChipName;
    params.pCounterAvailabilityImage = counters.data();
    if (!check_status(cuptiProfilerHostInitialize(&params))) {
      return false;
    }
    object_ = params.pHostObject;

    Metrics metrics{kMetric};
    CUpti_Profiler_Host_ConfigAddMetrics_Params add{
      CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE};
    add.pHostObject = object_;
    add.ppMetricNames = metrics.data();
    add.numMetrics = metrics.size();
    CUpti_Profiler_Host_GetConfigImageSize_Params size{
      CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE};
    size.pHostObject = object_;
    if (!check_status(cuptiProfilerHostConfigAddMetrics(&add)) ||
        !check_status(cuptiProfilerHostGetConfigImageSize(&size))) {
      return false;
    }
    config_.resize(size.configImageSize);
    CUpti_Profiler_Host_GetConfigImage_Params image{
      CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE};
    image.pHostObject = object_;
    image.pConfigImage = config_.data();
    image.configImageSize = config_.size();
    return check_status(cuptiProfilerHostGetConfigImage(&image));
  }

  [[nodiscard]] const Image& config() const {
    return config_;
  }

  [[nodiscard]] std::optional<double> evaluate(const Image& counters) const {
    CUpti_RangeProfiler_GetCounterDataInfo_Params info{
      CUpti_RangeProfiler_GetCounterDataInfo_Params_STRUCT_SIZE};
    info.pCounterDataImage = counters.data();
    info.counterDataImageSize = counters.size();
    if (!check_status(cuptiRangeProfilerGetCounterDataInfo(&info)) || info.numTotalRanges != 1) {
      return std::nullopt;
    }
    Metrics metrics{kMetric};
    double percent = 0.0;
    CUpti_Profiler_Host_EvaluateToGpuValues_Params params{
      CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE};
    params.pHostObject = object_;
    params.pCounterDataImage = counters.data();
    params.counterDataImageSize = counters.size();
    params.rangeIndex = 0;
    params.ppMetricNames = metrics.data();
    params.numMetrics = metrics.size();
    params.pMetricValues = &percent;
    if (!check_status(cuptiProfilerHostEvaluateToGpuValues(&params))) {
      return std::nullopt;
    }
    return percent / 100.0;
  }

private:
  CUpti_Profiler_Host_Object* object_ = nullptr;
  Image config_;
};

/// Builds the device's metric config once; nullptr when profiling is unavailable.
const Host* host_for(int device, CUcontext context) {
  static std::map<int, std::unique_ptr<Host>> hosts;
  if (const auto found = hosts.find(device); found != hosts.end()) {
    return found->second.get();
  }
  static const bool initialized = [] {
    CUpti_Profiler_Initialize_Params params{CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    return check_status(cuptiProfilerInitialize(&params));
  }();
  auto host = std::make_unique<Host>();
  if (!initialized || !host->init(device, context)) {
    HOST_LOG("Achieved occupancy unavailable on device {}", device);
    host.reset();
  }
  return hosts.emplace(device, std::move(host)).first->second.get();
}

// One user range around the launch; user replay reruns it once per pass.
bool collect_pass(
  CUpti_RangeProfiler_Object* object,
  cudaStream_t stream,
  const std::function<void(cudaStream_t)>& launch,
  bool& done
) {
  CUpti_RangeProfiler_Start_Params start{CUpti_RangeProfiler_Start_Params_STRUCT_SIZE};
  start.pRangeProfilerObject = object;
  CUpti_RangeProfiler_PushRange_Params push{CUpti_RangeProfiler_PushRange_Params_STRUCT_SIZE};
  push.pRangeProfilerObject = object;
  push.pRangeName = "launch";
  if (!check_status(cuptiRangeProfilerStart(&start)) ||
      !check_status(cuptiRangeProfilerPushRange(&push))) {
    return false;
  }
  launch(stream);
  const bool launched =
    check_status(cudaGetLastError()) && check_status(cudaStreamSynchronize(stream));
  CUpti_RangeProfiler_PopRange_Params pop{CUpti_RangeProfiler_PopRange_Params_STRUCT_SIZE};
  pop.pRangeProfilerObject = object;
  CUpti_RangeProfiler_Stop_Params stop{CUpti_RangeProfiler_Stop_Params_STRUCT_SIZE};
  stop.pRangeProfilerObject = object;
  const bool stopped =
    check_status(cuptiRangeProfilerPopRange(&pop)) && check_status(cuptiRangeProfilerStop(&stop));
  done = stop.isAllPassSubmitted != 0;
  return launched && stopped;
}

std::optional<double> collect(
  const Host& host,
  CUpti_RangeProfiler_Object* object,
  cudaStream_t stream,
  const std::function<void(cudaStream_t)>& launch
) {
  Metrics metrics{kMetric};
  CUpti_RangeProfiler_GetCounterDataSize_Params size{
    CUpti_RangeProfiler_GetCounterDataSize_Params_STRUCT_SIZE};
  size.pRangeProfilerObject = object;
  size.pMetricNames = metrics.data();
  size.numMetrics = metrics.size();
  size.maxNumOfRanges = 1;
  size.maxNumRangeTreeNodes = 1;
  if (!check_status(cuptiRangeProfilerGetCounterDataSize(&size))) {
    return std::nullopt;
  }
  Image counters(size.counterDataSize);
  CUpti_RangeProfiler_CounterDataImage_Initialize_Params image{
    CUpti_RangeProfiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
  image.pRangeProfilerObject = object;
  image.pCounterData = counters.data();
  image.counterDataSize = counters.size();

  CUpti_RangeProfiler_SetConfig_Params config{CUpti_RangeProfiler_SetConfig_Params_STRUCT_SIZE};
  config.pRangeProfilerObject = object;
  config.pConfig = host.config().data();
  config.configSize = host.config().size();
  config.pCounterDataImage = counters.data();
  config.counterDataImageSize = counters.size();
  config.range = CUPTI_UserRange;
  config.replayMode = CUPTI_UserReplay;
  config.maxRangesPerPass = 1;
  config.numNestingLevels = 1;
  config.minNestingLevel = 1;
  config.passIndex = 0;
  config.targetNestingLevel = 1;
  if (!check_status(cuptiRangeProfilerCounterDataImageInitialize(&image)) ||
      !check_status(cuptiRangeProfilerSetConfig(&config))) {
    return std::nullopt;
  }

  bool done = false;
  for (int pass = 0; !done; ++pass) {
    if (pass == kMaxPasses || !collect_pass(object, stream, launch, done)) {
      return std::nullopt;
    }
  }
  CUpti_RangeProfiler_DecodeData_Params decode{CUpti_RangeProfiler_DecodeData_Params_STRUCT_SIZE};
  decode.pRangeProfilerObject = object;
  if (!check_status(cuptiRangeProfilerDecodeData(&decode))) {
    return std::nullopt;
  }
  return host.evaluate(counters);
}

} // namespace

std::optional<double> achieved_occupancy(
  int device, cudaStream_t stream, const std::function<void(cudaStream_t)>& launch
) {
  // Nsight tools inject their own CUPTI session, which owns the counters.
  if (std::getenv("CUDA_INJECTION64_PATH") != nullptr) {
    return std::nullopt;
  }
  CUcontext context = nullptr;
  if (!check_status(cudaSetDevice(device)) || !check_status(cuCtxGetCurrent(&context))) {
    return std::nullopt;
  }
  const Host* host = host_for(device, context);
  if (host == nullptr) {
    return std::nullopt;
  }
  CUpti_RangeProfiler_Enable_Params enable{CUpti_RangeProfiler_Enable_Params_STRUCT_SIZE};
  enable.ctx = context;
  if (!check_status(cuptiRangeProfilerEnable(&enable))) {
    return std::nullopt;
  }
  const auto value = collect(*host, enable.pRangeProfilerObject, stream, launch);
  CUpti_RangeProfiler_Disable_Params disable{CUpti_RangeProfiler_Disable_Params_STRUCT_SIZE};
  disable.pRangeProfilerObject = enable.pRangeProfilerObject;
  check_status(cuptiRangeProfilerDisable(&disable));
  return value;
}

} // namespace lg::benchmark
