#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace lg::matmul {

// Every dimension is validated to fit in an int, so on a 64-bit size_t any
// product of two of them fits without overflow.
inline constexpr auto kMaxDimension = static_cast<unsigned>(std::numeric_limits<int>::max());
static_assert(sizeof(std::size_t) >= 8, "dimension products assume a 64-bit size_t");

/// Row-major C[m,n] = A[m,k] * B[k,n] extents.
class GemmShape {
public:
  /// Creates validated GEMM dimensions.
  GemmShape(unsigned m, unsigned n, unsigned k);

  /// Returns the output row count.
  [[nodiscard]] unsigned m() const {
    return m_;
  }
  /// Returns the output column count.
  [[nodiscard]] unsigned n() const {
    return n_;
  }
  /// Returns the reduction dimension.
  [[nodiscard]] unsigned k() const {
    return k_;
  }
  /// Returns the left matrix size.
  [[nodiscard]] std::size_t a_size() const {
    return std::size_t{m_} * k_;
  }
  /// Returns the right matrix size.
  [[nodiscard]] std::size_t b_size() const {
    return std::size_t{k_} * n_;
  }
  /// Returns the output matrix size.
  [[nodiscard]] std::size_t c_size() const {
    return std::size_t{m_} * n_;
  }

private:
  unsigned m_;
  unsigned n_;
  unsigned k_;
};

class Matrix {
public:
  /// Creates a zero-filled row-major matrix.
  Matrix(unsigned rows, unsigned cols);

  /// Returns the row count.
  [[nodiscard]] unsigned rows() const {
    return rows_;
  }
  /// Returns the column count.
  [[nodiscard]] unsigned cols() const {
    return cols_;
  }
  /// Returns the element count.
  [[nodiscard]] std::size_t size() const {
    return values_.size();
  }
  /// Returns the storage size in bytes.
  [[nodiscard]] std::size_t bytes() const {
    return size() * sizeof(float);
  }
  /// Returns mutable contiguous storage.
  [[nodiscard]] float* data() {
    return values_.data();
  }
  /// Returns immutable contiguous storage.
  [[nodiscard]] const float* data() const {
    return values_.data();
  }

  /// Returns a bounds-checked mutable element.
  float& operator()(unsigned row, unsigned col);
  /// Returns a bounds-checked immutable element.
  const float& operator()(unsigned row, unsigned col) const;

  /// Returns the mutable begin iterator.
  [[nodiscard]] auto begin() {
    return values_.begin();
  }
  /// Returns the mutable end iterator.
  [[nodiscard]] auto end() {
    return values_.end();
  }
  /// Returns the immutable begin iterator.
  [[nodiscard]] auto begin() const {
    return values_.begin();
  }
  /// Returns the immutable end iterator.
  [[nodiscard]] auto end() const {
    return values_.end();
  }

private:
  unsigned rows_;
  unsigned cols_;
  std::vector<float> values_;
};

struct Problem {
  /// Allocates matrices for one GEMM problem.
  explicit Problem(GemmShape shape);

  GemmShape shape;
  Matrix a;
  Matrix b;
  Matrix expected;
  Matrix result;
};

struct RandomFill {
  std::uint32_t seed = 0x4D41544DU;
  float minimum = -1.0F;
  float maximum = 1.0F;
};

/// Fills one matrix deterministically.
void fill_random(Matrix& matrix, RandomFill config = {});
/// Fills both input matrices deterministically.
void fill_random(Problem& problem, RandomFill config = {});

} // namespace lg::matmul
