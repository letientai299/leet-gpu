#pragma once

#include "matmul/kernel.hpp"

void launch_matmul_cell(const float* a,
                        const float* b,
                        float* c,
                        const lg::matmul::GemmShape& shape,
                        cudaStream_t stream = nullptr);
void launch_matmul_row(const float* a,
                       const float* b,
                       float* c,
                       const lg::matmul::GemmShape& shape,
                       cudaStream_t stream = nullptr);
void launch_matmul_col(const float* a,
                       const float* b,
                       float* c,
                       const lg::matmul::GemmShape& shape,
                       cudaStream_t stream = nullptr);

inline const lg::matmul::Kernel kMatmulCell{"matmul.cell", launch_matmul_cell};
inline const lg::matmul::Kernel kMatmulRow{"matmul.row", launch_matmul_row};
inline const lg::matmul::Kernel kMatmulCol{"matmul.col", launch_matmul_col};
