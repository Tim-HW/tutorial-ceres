//
// Author: keir@google.com (Keir Mierle)
//
// Minimize 0.5 (10 - x)^2 using jacobian matrix computed using
// numeric differentiation.
#include "ceres/ceres.h"
#include "glog/logging.h"
// A cost functor that implements the residual r = 10 - x.
struct CostFunctor {
  bool operator()(const double* const x, double* residual) const {
    residual[0] = 10.0 - x[0];
    return true;
  }
};
int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  // The variable to solve for with its initial value. It will be
  // mutated in place by the solver.
  double x = 0.5;
  const double initial_x = x;
  // Build the problem.
  ceres::Problem problem;
  // Set up the only cost function (also known as residual). This uses
  // numeric differentiation to obtain the derivative (jacobian).
  ceres::CostFunction* cost_function =
      new ceres::NumericDiffCostFunction<CostFunctor, ceres::CENTRAL, 1, 1>(
          new CostFunctor);

  problem.AddResidualBlock(cost_function, nullptr, &x);
  
  // Run the solver!
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  
  std::cout << summary.BriefReport() << "\n";
  std::cout << "x : " << initial_x << " -> " << x << "\n";
  
  return 0;
}