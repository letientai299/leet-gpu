#include "benchmark/format.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

using lg::benchmark::Run;

void require(bool condition, const std::string& what) {
  if (!condition) {
    throw std::runtime_error("format regression failed: " + what);
  }
}

std::string number(const char* tag, const char* name, const std::string& value, const char* hint) {
  return std::string(R"({"tag":")") + tag + R"(","name":")" + name + R"(","hint":")" + hint +
         R"(","data":[{"name":"value","type":"float64","value":")" + value + R"("}]})";
}

std::string text(const char* tag, const char* name, const char* value) {
  return std::string(R"({"tag":")") + tag + R"(","name":")" + name +
         R"(","data":[{"name":"value","type":"string","value":")" + value + R"("}]})";
}

struct Spec {
  std::string seconds = "0.001";
  std::string noise = "0.001";
  std::string rate = "100e9";
  std::string status = "stable";
  std::string axes = "[]";
  std::vector<std::string> extras;
  int states = 1;
};

std::string run_json(const Spec& spec) {
  std::string summaries =
    number("nv/cold/time/gpu/mean", "GPU", spec.seconds, "duration") + "," +
    number("nv/cold/time/gpu/stdev/relative", "Noise", spec.noise, "percentage") + "," +
    number("nv/cold/time/gpu/min", "Min", "0.0009", "duration") + "," +
    text("lg/cold/status", "Status", spec.status.c_str());
  if (!spec.rate.empty()) {
    summaries += "," + number("lg/throughput", "Elem/s", spec.rate, "item_rate");
  }
  for (const auto& extra : spec.extras) {
    summaries += "," + extra;
  }
  std::string states;
  for (int state = 0; state < spec.states; ++state) {
    states += std::string(state == 0 ? "" : ",") + R"({"device":0,"axis_values":)" + spec.axes +
              R"(,"summaries":[)" + summaries + "]}";
  }
  return R"({"devices":[{"id":0,"name":"GPU"}],"benchmarks":[{"name":"bench","states":[)" + states +
         "]}]}";
}

// Stdout is redirected to a file for the call and restored afterwards.
class Capture {
public:
  Capture() {
    std::fflush(stdout);
    saved_ = dup(fileno(stdout));
    require(saved_ >= 0, "dup");
    path_ = "/tmp/leet-gpu-format-XXXXXX";
    const int descriptor = mkstemp(path_.data());
    require(descriptor >= 0, "mkstemp");
    close(descriptor);
    require(std::freopen(path_.c_str(), "w", stdout) != nullptr, "freopen");
  }
  Capture(const Capture&) = delete;
  Capture& operator=(const Capture&) = delete;
  ~Capture() {
    std::fflush(stdout);
    dup2(saved_, fileno(stdout));
    close(saved_);
    std::remove(path_.c_str());
  }

  [[nodiscard]] std::string output() const {
    std::fflush(stdout);
    const std::ifstream file(path_);
    std::stringstream content;
    content << file.rdbuf();
    return content.str();
  }

private:
  int saved_ = -1;
  std::string path_;
};

std::string compare(const std::vector<Run>& runs) {
  const Capture capture;
  lg::benchmark::print_comparison(runs);
  return capture.output();
}

std::string single(const std::string& json) {
  const Capture capture;
  lg::benchmark::print_result(json);
  return capture.output();
}

std::string plain(const std::string& text) {
  std::string out;
  for (std::size_t index = 0; index < text.size(); ++index) {
    if (text[index] == '\033') {
      index = text.find('m', index);
    } else {
      out += text[index];
    }
  }
  return out;
}

struct Case {
  const char* name;
  std::vector<Run> runs;
  std::vector<const char*> plain_parts;
  std::vector<const char*> raw_parts;
};

void check_cases() {
  const std::string noisy = "0.03";
  const std::vector<Case> cases{
    {"ratios",
     {{"base", run_json({})}, {"fast", run_json({.seconds = "0.0005", .rate = "200e9"})}},
     {"base 1.00 ms", "fast 500.00 us 0.50×", "2.00×", "all  stable", "min 900.00 us",
      "min 0.90 ms"},
     {}},
    {"direction",
     {{"base", run_json({})},
      {"fast", run_json({.seconds = "0.0005", .rate = "200e9"})},
      {"slow", run_json({.seconds = "0.002", .rate = "50e9"})}},
     {},
     {"\033[32m0.50×", "\033[32m2.00×", "\033[31m2.00×", "\033[31m0.50×"}},
    {"noise band",
     {{"base", run_json({.seconds = "0.001", .noise = noisy})},
      {"near", run_json({.seconds = "0.00098", .noise = noisy})}},
     {"0.98×"},
     {"\033[2m0.98×"}},
    {"missing row", {{"base", run_json({})}, {"bare", run_json({.rate = ""})}}, {"bare n/a"}, {}},
    {"missing baseline row",
     {{"bare", run_json({.rate = ""})}, {"full", run_json({})}},
     {"bare n/a", "full 100.0 G/s     –"},
     {}},
    {"duplicate labels",
     {{"a", run_json({.extras = {number("x/1", "Zed", "5", ""), number("x/2", "Zed", "6", "")}})},
      {"b", run_json({.extras = {number("x/1", "Zed", "5", ""), number("x/2", "Zed", "6", "")}})}},
     {"b   6         1.00×"},
     {}},
    {"three apps",
     {{"a", run_json({})},
      {"b", run_json({.seconds = "0.0005"})},
      {"c", run_json({.seconds = "0.002"})}},
     {"0.50×", "2.00×"},
     {}},
    {"differing status",
     {{"a", run_json({.status = "stable"})}, {"b", run_json({.status = "UNSTABLE"})}},
     {"a   stable", "b   UNSTABLE"},
     {}},
  };
  for (const auto& test : cases) {
    const std::string output = compare(test.runs);
    for (const char* part : test.plain_parts) {
      require(
        plain(output).find(part) != std::string::npos,
        std::string(test.name) + ": " + part + "\n" + plain(output)
      );
    }
    for (const char* part : test.raw_parts) {
      require(output.find(part) != std::string::npos, std::string(test.name) + ": raw " + part);
    }
  }
}

template <typename Exception> void rejects(const std::vector<Run>& runs, const char* name) {
  bool rejected = false;
  try {
    const Capture capture;
    lg::benchmark::print_comparison(runs);
  } catch (const Exception&) {
    rejected = true;
  }
  require(rejected, name);
}

void check_errors() {
  rejects<std::invalid_argument>({{"only", run_json({})}}, "one run");
  rejects<std::runtime_error>(
    {{"a", run_json({.states = 2})}, {"b", run_json({.states = 1})}}, "state count"
  );
  rejects<std::runtime_error>(
    {{"a", run_json({.axes = R"([{"name":"N","value":"1"}])"})}, {"b", run_json({})}}, "axes"
  );
  // Nothing prints before a mismatch is found.
  const Capture capture;
  try {
    lg::benchmark::print_comparison(
      std::vector<Run>{{"a", run_json({.states = 2})}, {"b", run_json({.states = 1})}}
    );
  } catch (const std::runtime_error&) {
    require(capture.output().empty(), "partial output");
    return;
  }
  require(false, "no mismatch error");
}

void check_single() {
  const std::string output = plain(single(run_json({})));
  require(
    output.find(
      " measured  cold: L2 flushed per run · batch: back-to-back\n"
      "  Cold       1.00 ms ±0.10%  min 0.90\n"
      "  Status     stable\n"
      "  Elem/s     100.0 G/s\n"
    ) != std::string::npos,
    "single-run rows"
  );
}

} // namespace

int main() try {
  // Colors must be on before the first paint to pin ratio tints.
  unsetenv("NO_COLOR");
  setenv("FORCE_COLOR", "1", 1);
  check_cases();
  check_errors();
  check_single();
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
