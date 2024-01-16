
#include "ceres/ceres.h"
#include "glog/logging.h"


// A simple example of using the Ceres minimizer.
//
// Minimize 0.5 (10 - x)^2 using jacobian matrix computed using
// automatic differentiation.

using namespace ceres;

struct F1 {
  template <typename T>
  bool operator()(const T* const x1, const T* const x2, T* residual) const {
    residual[0] = x1[0] + 10.0*x2[0];
    return true;
  }
};

struct F2 {
  template <typename T>
  bool operator()(const T* const x3, const T* const x4, T* residual) const {
    residual[0] = sqrt(5.0) * (x3[0] - x4[0]);
    return true;
  }
};

struct F3 {
  template <typename T>
  bool operator()(const T* const x2, const T* const x3, T* residual) const {
    residual[0] = (x2[0] - 2.0*x3[0]) * (x2[0] - 2.0*x3[0]);
    return true;
  }
};

struct F4 {
  template <typename T>
  bool operator()(const T* const x1, const T* const x4, T* residual) const {
    residual[0] = sqrt(10.0) * (x1[0] - x4[0]) * (x1[0] - x4[0]);
    return true;
  }
};


DEFINE_string(minimizer,
              "trust_region",
              "Minimizer type to use, choices are: line_search & trust_region");



int main(int argc, char** argv) {
  
  google::InitGoogleLogging(argv[0]);

    // The variable to solve for with its initial value.
  double x1 =  3.0; double x2 = -1.0; double x3 =  0.0; double x4 = 1.0;

  Problem problem;

  // Add residual terms to the problem using the autodiff
  // wrapper to get the derivatives automatically.
  problem.AddResidualBlock(
    new AutoDiffCostFunction<F1, 1, 1, 1>(new F1), nullptr, &x1, &x2);
  problem.AddResidualBlock(
    new AutoDiffCostFunction<F2, 1, 1, 1>(new F2), nullptr, &x3, &x4);
  problem.AddResidualBlock(
    new AutoDiffCostFunction<F3, 1, 1, 1>(new F3), nullptr, &x2, &x3);
  problem.AddResidualBlock(
    new AutoDiffCostFunction<F4, 1, 1, 1>(new F4), nullptr, &x1, &x4);

  ceres::Solver::Options options;
  
  LOG_IF(FATAL,
          !ceres::StringToMinimizerType(CERES_GET_FLAG(FLAGS_minimizer),
                                        &options.minimizer_type))
        << "Invalid minimizer: " << CERES_GET_FLAG(FLAGS_minimizer)
        << ", valid options are: trust_region and line_search.";
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  // clang-format off
  std::cout << "Initial x1 = " << x1
            << ", x2 = " << x2
            << ", x3 = " << x3
            << ", x4 = " << x4
            << "\n";
  // clang-format on
  // Run the solver!
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";
  // clang-format off
  std::cout << "Final x1 = " << x1
            << ", x2 = " << x2
            << ", x3 = " << x3
            << ", x4 = " << x4
            << "\n";
  // clang-format on
  return 0;
}