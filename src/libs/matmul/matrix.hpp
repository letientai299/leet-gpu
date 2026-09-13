#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lg::matmul {

class GemmShape {
public:
  GemmShape(unsigned m, unsigned n, unsigned k);

  [[nodiscard]] unsigned m() const {
    return m_;
  }
  [[nodiscard]] unsigned n() const {
    return n_;
  }
  [[nodiscard]] unsigned k() const {
    return k_;
  }
  [[nodiscard]] std::size_t a_size() const;
  [[nodiscard]] std::size_t b_size() const;
  [[nodiscard]] std::size_t c_size() const;

private:
  unsigned m_;
  unsigned n_;
  unsigned k_;
};

class Matrix {
public:
  Matrix(unsigned rows, unsigned cols);

  [[nodiscard]] unsigned rows() const {
    return rows_;
  }
  [[nodiscard]] unsigned cols() const {
    return cols_;
  }
  [[nodiscard]] std::size_t size() const {
    return values_.size();
  }
  [[nodiscard]] std::size_t bytes() const {
    return size() * sizeof(float);
  }
  [[nodiscard]] float* data() {
    return values_.data();
  }
  [[nodiscard]] const float* data() const {
    return values_.data();
  }

  [[nodiscard]] float& operator()(unsigned row, unsigned col);
  [[nodiscard]] const float& operator()(unsigned row, unsigned col) const;

  [[nodiscard]] auto begin() {
    return values_.begin();
  }
  [[nodiscard]] auto end() {
    return values_.end();
  }
  [[nodiscard]] auto begin() const {
    return values_.begin();
  }
  [[nodiscard]] auto end() const {
    return values_.end();
  }

private:
  unsigned rows_;
  unsigned cols_;
  std::vector<float> values_;
};

struct Problem {
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

void fill_random(Matrix& matrix, RandomFill config = {});
void fill_random(Problem& problem, RandomFill config = {});

} // namespace lg::matmul
