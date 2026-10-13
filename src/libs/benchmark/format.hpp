#pragma once

#include <string_view>

namespace lg::benchmark {

/// Prints captured NVBench JSON as grouped rows.
void print_result(std::string_view input);

} // namespace lg::benchmark
