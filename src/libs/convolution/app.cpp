#include "convolution/app.hpp"

#include "checks.hpp"
#include "convolution/correctness.hpp"

namespace lg::convolution {

int run_check(int argc, char** argv, Kernel kernel, Shape shape) {
  return run_host(argc, argv, [&] {
    Problem problem{shape, {}, {}, {}, {}};
    fill_problem(problem);
    return check(problem, kernel);
  });
}

} // namespace lg::convolution
