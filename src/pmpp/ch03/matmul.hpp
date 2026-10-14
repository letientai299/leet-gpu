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

inline const lg::matmul::Kernel kMatmulCell{.name = "matmul.cell",
                                            .launch = launch_matmul_cell,
                                            .traffic = lg::matmul::cell_traffic,
                                            .resources = matmul_cell_resources};
inline const lg::matmul::Kernel kMatmulRow{.name = "matmul.row",
                                           .launch = launch_matmul_row,
                                           .traffic = lg::matmul::cell_traffic,
                                           .resources = matmul_row_resources};
inline const lg::matmul::Kernel kMatmulCol{.name = "matmul.col",
                                           .launch = launch_matmul_col,
                                           .traffic = lg::matmul::cell_traffic,
                                           .resources = matmul_col_resources};
