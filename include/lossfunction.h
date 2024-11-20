#pragma once

#include <cmath>
#include <vector>

class FinanceLossFunctions {
public:
  FinanceLossFunctions() = default;

  double computeLoss(const std::vector<double> &predictedPrices,
                     const std::vector<double> &marketPrices,
                     const std::vector<std::vector<double>> &features);

private:
  double meanSquaredError(const std::vector<double> &predicted,
                          const std::vector<double> &actual) const;
  double arbitragePenalty(const std::vector<double> &predictedPrices) const;
};
