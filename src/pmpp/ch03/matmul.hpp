#pragma once

#include "matmul/kernel.hpp"

void launch_matmul_cell(
  const float* a,
  const float* b,
  float* c,
  const lg::matmul::GemmShape& shape,
  cudaStream_t stream = nullptr
);
void launch_matmul_row(
  const float* a,
  const float* b,
  float* c,
  const lg::matmul::GemmShape& shape,
  cudaStream_t stream = nullptr
);
void launch_matmul_col(
  const float* a,
  const float* b,
  float* c,
  const lg::matmul::GemmShape& shape,
  cudaStream_t stream = nullptr
);

cudaError_t
matmul_cell_resources(const lg::matmul::GemmShape&, lg::benchmark::KernelResources& resources);
cudaError_t
matmul_row_resources(const lg::matmul::GemmShape&, lg::benchmark::KernelResources& resources);
cudaError_t
matmul_col_resources(const lg::matmul::GemmShape&, lg::benchmark::KernelResources& resources);

inline const lg::matmul::Kernel kMatmulCell{
  "matmul.cell", launch_matmul_cell, lg::matmul::cell_traffic, {}, matmul_cell_resources};
inline const lg::matmul::Kernel kMatmulRow{
  "matmul.row", launch_matmul_row, lg::matmul::cell_traffic, {}, matmul_row_resources};
inline const lg::matmul::Kernel kMatmulCol{
  "matmul.col", launch_matmul_col, lg::matmul::cell_traffic, {}, matmul_col_resources};
