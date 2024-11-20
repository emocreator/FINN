#include "../include/lossfunction.h"

double FinanceLossFunctions::computeLoss(
    const std::vector<double> &predictedPrices,
    const std::vector<double> &marketPrices,
    const std::vector<std::vector<double>> &features) {
  double mse = meanSquaredError(predictedPrices, marketPrices);
  double arbitragePenaltyValue = arbitragePenalty(predictedPrices);

  // Total loss is MSE plus arbitrage penalty
  return mse + arbitragePenaltyValue;
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
