#include "matmul/matrix.hpp"

#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>

namespace lg::matmul {
namespace {

std::size_t checked_size(unsigned rows, unsigned cols) {
  if (rows == 0 || cols == 0) {
    throw std::invalid_argument("matrix dimensions must be positive");
  }
  if (rows > std::numeric_limits<std::size_t>::max() / cols) {
    throw std::length_error("matrix size overflow");
  }
  return static_cast<std::size_t>(rows) * cols;
}

void validate_fill(RandomFill config) {
  if (config.minimum > config.maximum) {
    throw std::invalid_argument("random range is reversed");
  }
}

} // namespace

GemmShape::GemmShape(unsigned m, unsigned n, unsigned k) : m_(m), n_(n), k_(k) {
  constexpr auto maximum = static_cast<unsigned>(std::numeric_limits<int>::max());
  if (m > maximum || n > maximum || k > maximum) {
    throw std::invalid_argument("matrix dimensions exceed INT_MAX");
  }
  (void)a_size();
  (void)b_size();
  (void)c_size();
}

std::size_t GemmShape::a_size() const {
  return checked_size(m_, k_);
}

std::size_t GemmShape::b_size() const {
  return checked_size(k_, n_);
}

std::size_t GemmShape::c_size() const {
  return checked_size(m_, n_);
}

Matrix::Matrix(unsigned rows, unsigned cols)
    : rows_(rows), cols_(cols), values_(checked_size(rows, cols)) {
}

float& Matrix::operator()(unsigned row, unsigned col) {
  return values_.at(static_cast<std::size_t>(row) * cols_ + col);
}

const float& Matrix::operator()(unsigned row, unsigned col) const {
  return values_.at(static_cast<std::size_t>(row) * cols_ + col);
}

Problem::Problem(GemmShape problem_shape)
    : shape(problem_shape), a(shape.m(), shape.k()), b(shape.k(), shape.n()),
      expected(shape.m(), shape.n()), result(shape.m(), shape.n()) {
}

void fill_random(Matrix& matrix, RandomFill config) {
  validate_fill(config);
  std::mt19937 rng(config.seed); // NOLINT(bugprone-random-generator-seed)
  std::uniform_real_distribution<float> dist(config.minimum, config.maximum);
  std::generate(matrix.begin(), matrix.end(), [&] {
    return dist(rng);
  });
}

void fill_random(Problem& problem, RandomFill config) {
  validate_fill(config);
  std::mt19937 rng(config.seed); // NOLINT(bugprone-random-generator-seed)
  std::uniform_real_distribution<float> dist(config.minimum, config.maximum);
  const auto next = [&] {
    return dist(rng);
  };
  std::generate(problem.a.begin(), problem.a.end(), next);
  std::generate(problem.b.begin(), problem.b.end(), next);
}

} // namespace lg::matmul
