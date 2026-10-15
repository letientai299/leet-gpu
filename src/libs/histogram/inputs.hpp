#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace lg::histogram {

inline constexpr std::size_t kInputBytes = 21651042;
inline constexpr std::uint64_t kInputChecksum = 10832120231796306158ULL;

struct Input {
  std::filesystem::path path;
  std::size_t size = kInputBytes;
  std::uint64_t checksum = kInputChecksum;
  std::array<unsigned int, 7> bins{2486624, 3478949, 1754813, 2914974, 3279925, 986783, 334467};
};

std::filesystem::path input_directory();
Input prepare_input(const std::filesystem::path& directory);
std::string read_input(const Input& input);

} // namespace lg::histogram
