#include <chrono>
#include <ifopt/constraint_set.h>
#include <ifopt/cost_term.h>
#include <ifopt/ipopt_solver.h>
#include <ifopt/problem.h>
#include <ifopt/variable_set.h>
#include <iostream>

/*
  x in R^2 x = [x0, x1]

  min sqrt(x1)
  s.t.
  x1 >= 0
  x1 - (-x0 + 1)^3 >= 0
  x1 - (2x0)^3 >= 0

  ans: 0.54433 (x0, x1) = (1/3, 8/27)

*/

class MyVariables : public ifopt::VariableSet
{
public:
  MyVariables() : MyVariables("var_set1") {};
  MyVariables(const std::string& name) : ifopt::VariableSet(2, name)
  {
    // the initial values where the NLP starts iterating from
    x0_ = 1.234;
    x1_ = 5.678;
  }

  void SetVariables(const VectorXd& x) override
  {
    x0_ = x(0);
    x1_ = x(1);
  }

  Eigen::VectorXd GetValues() const override
  {
    Eigen::VectorXd ret = Eigen::VectorXd::Zero(2);
    ret << x0_, x1_;
    return ret;
  }

  std::vector<ifopt::Bounds> GetBounds() const override
  {
    std::vector<ifopt::Bounds> bounds(GetRows());
    bounds.at(0) = ifopt::NoBound;
    bounds.at(1) = ifopt::BoundGreaterZero;
    return bounds;
  }

private:
  double x0_, x1_;
};

class MyCost : public ifopt::CostTerm
{
public:
  MyCost() : MyCost("cost_term1") {}
  MyCost(const std::string& name) : CostTerm(name) {}

  double GetCost() const override
  {
    Eigen::VectorXd x = GetVariables()->GetComponent("var_set1")->GetValues();
    return std::sqrt(x(1));
  }

  void FillJacobianBlock (std::string var_set, Jacobian& jac) const override
  {
    if (var_set == "var_set1") {
      Eigen::VectorXd x = GetVariables()->GetComponent("var_set1")->GetValues();

      jac.coeffRef(0, 0) = 0.0;                   // derivative of cost w.r.t x0
      jac.coeffRef(0, 1) = 0.5 / std::sqrt(x(1)); // derivative of cost w.r.t x1
    }
  }
};

class MyConstraint : public ifopt::ConstraintSet
{
public:
  MyConstraint() : MyConstraint("constraint1") {}
  MyConstraint(const std::string& name) : ConstraintSet(3, name) {}

  Eigen::VectorXd GetValues() const override
  {
    Eigen::VectorXd g(GetRows());
    Eigen::VectorXd x = GetVariables()->GetComponent("var_set1")->GetValues();
    g(0) = x(1);
    g(1) = x(1) - (-x(0) + 1) * (-x(0) + 1) * (-x(0) + 1);
    g(2) = x(1) - (2 * x(0)) * (2 * x(0)) * (2 * x(0));
    return g;
  }

  std::vector<ifopt::Bounds> GetBounds() const override
  {
    std::vector<ifopt::Bounds> b(GetRows());
    b.at(0) = ifopt::BoundGreaterZero;
    b.at(1) = ifopt::BoundGreaterZero;
    b.at(2) = ifopt::BoundGreaterZero;
    return b;
  }

  void FillJacobianBlock (std::string var_set, Eigen::SparseMatrix<double, Eigen::RowMajor>& jac_block) const override
  {
    if(var_set == "var_set1") {
      Eigen::VectorXd x = GetVariables()->GetComponent("var_set1")->GetValues();

      jac_block.coeffRef(0, 0) = 0;
      jac_block.coeffRef(0, 1) = 1;
      jac_block.coeffRef(1, 0) = 3 * (-x(0) + 1) * (-x(0) + 1);
      jac_block.coeffRef(1, 1) = 1;
      jac_block.coeffRef(2, 0) = -2 * 3 * (2 * x(0)) * (2 * x(0));
      jac_block.coeffRef(2, 1) = 1;
    }
  }
};

int main()
{
  ifopt::Problem nlp;
  nlp.AddVariableSet(std::make_shared<MyVariables>());
  nlp.AddConstraintSet(std::make_shared<MyConstraint>());
  nlp.AddCostSet(std::make_shared<MyCost>());
  // nlp.PrintCurrent();

  ifopt::IpoptSolver ipopt;
  ipopt.SetOption("linear_solver", "mumps");
  ipopt.SetOption("jacobian_approximation", "exact");
  ipopt.SetOption("print_level", 0);
  ipopt.SetOption("sb", "yes");

  auto start = std::chrono::high_resolution_clock::now();
  ipopt.Solve(nlp);
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  Eigen::VectorXd x = nlp.GetOptVariables()->GetValues();
  double opt_x[x.size()];
  opt_x[0] = x[0];
  opt_x[1] = x[1];
  double min_val = nlp.EvaluateCostFunction(opt_x);
  std::cout << "solved with " << nlp.GetIterationCount() << " iteration" << std::endl;
  std::cout << "final cost: " << min_val << std::endl;
  std::cout << "final solution: " << x.transpose() << std::endl;
  std::cout << "solve time: " << duration << "[us]"<< std::endl;
}
