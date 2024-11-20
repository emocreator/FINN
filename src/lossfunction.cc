#include "../include/lossfunction.h"
#include <stdexcept>

double FinanceLossFunctions::computeLoss(
    const std::vector<double> &predictedPrices,
    const std::vector<double> &marketPrices,
    [[maybe_unused]] const std::vector<std::vector<double>> &features) {

  if (predictedPrices.size() != marketPrices.size()) {
    throw std::invalid_argument(
        "Size mismatch between predicted and actual prices");
  }

  const double mse = meanSquaredError(predictedPrices, marketPrices);
  const double arbitragePenaltyValue = arbitragePenalty(predictedPrices);

  // Use weighted sum for better balance
  constexpr double mseFactor = 0.7;
  constexpr double arbitrageFactor = 0.3;

  return mseFactor * mse + arbitrageFactor * arbitragePenaltyValue;
}

double FinanceLossFunctions::meanSquaredError(
    const std::vector<double> &predicted,
    const std::vector<double> &actual) const {
  double mse = 0.0;
  for (size_t i = 0; i < predicted.size(); ++i) {
    double error = predicted[i] - actual[i];
    mse += error * error;
  }
  return mse / predicted.size();
}

double FinanceLossFunctions::arbitragePenalty(
    const std::vector<double> &predictedPrices) const {
  double penalty = 0.0;
  const double transactionCost = 0.001; // Example transaction cost
  for (size_t i = 0; i < predictedPrices.size() - 1; ++i) {
    double profit =
        predictedPrices[i + 1] - predictedPrices[i] - transactionCost;
    if (profit > 0) {
      penalty += profit * profit;
    }
  }
  return penalty / (predictedPrices.size() - 1);
}
