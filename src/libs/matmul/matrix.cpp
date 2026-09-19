#include "matmul/matrix.hpp"

#include <algorithm>
#include <initializer_list>
#include <random>
#include <stdexcept>

namespace lg::matmul {
namespace {

void validate_dimensions(std::initializer_list<unsigned> dimensions) {
  for (const unsigned dimension : dimensions) {
    if (dimension == 0) {
      throw std::invalid_argument("matrix dimensions must be positive");
    }
    if (dimension > kMaxDimension) {
      throw std::invalid_argument("matrix dimensions exceed INT_MAX");
    }
  }
}

/// Deterministic uniform floats; every matrix of a problem draws from one stream.
class RandomValues {
public:
  // rng_ initializes first, so validate() runs before dist_ sees the range.
  explicit RandomValues(RandomFill config)
      : rng_(validate(config).seed), dist_(config.minimum, config.maximum) {
  }

  void fill(Matrix& matrix) {
    std::generate(matrix.begin(), matrix.end(), [this] {
      return dist_(rng_);
    });
  }

private:
  static RandomFill validate(RandomFill config) {
    if (config.minimum > config.maximum) {
      throw std::invalid_argument("random range is reversed");
    }
    return config;
  }

  std::mt19937 rng_; // NOLINT(bugprone-random-generator-seed)
  std::uniform_real_distribution<float> dist_;
};

} // namespace

GemmShape::GemmShape(unsigned m, unsigned n, unsigned k) : m_(m), n_(n), k_(k) {
  validate_dimensions({m, n, k});
}

Matrix::Matrix(unsigned rows, unsigned cols) : rows_(rows), cols_(cols) {
  validate_dimensions({rows, cols});
  values_.assign(std::size_t{rows} * cols, 0.0F);
}

float& Matrix::operator()(unsigned row, unsigned col) {
  return values_.at(std::size_t{row} * cols_ + col);
}

const float& Matrix::operator()(unsigned row, unsigned col) const {
  return values_.at(std::size_t{row} * cols_ + col);
}

Problem::Problem(GemmShape problem_shape)
    : shape(problem_shape), a(shape.m(), shape.k()), b(shape.k(), shape.n()),
      expected(shape.m(), shape.n()), result(shape.m(), shape.n()) {
}

void fill_random(Matrix& matrix, RandomFill config) {
  RandomValues(config).fill(matrix);
}

void fill_random(Problem& problem, RandomFill config) {
  RandomValues values(config);
  values.fill(problem.a);
  values.fill(problem.b);
}

} // namespace lg::matmul
