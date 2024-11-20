#include "../include/utils.h"

void DataUtils::normalize(std::vector<std::vector<double>> &data) {
  if (data.empty())
    return;

  size_t numFeatures = data[0].size();
  std::vector<double> minVals(numFeatures, std::numeric_limits<double>::max());
  std::vector<double> maxVals(numFeatures,
                              std::numeric_limits<double>::lowest());

  // Find min and max for each feature
  for (const auto &row : data) {
    for (size_t i = 0; i < numFeatures; ++i) {
      minVals[i] = std::min(minVals[i], row[i]);
      maxVals[i] = std::max(maxVals[i], row[i]);
    }
  }

  // Normalize data
  for (auto &row : data) {
    for (size_t i = 0; i < numFeatures; ++i) {
      row[i] = (row[i] - minVals[i]) / (maxVals[i] - minVals[i] + 1e-8);
    }
  }
}

void DataUtils::normalize(std::vector<double> &data) {
  if (data.empty())
    return;

  double minVal = *std::min_element(data.begin(), data.end());
  double maxVal = *std::max_element(data.begin(), data.end());

  // Normalize data
  for (auto &val : data) {
    val = (val - minVal) / (maxVal - minVal + 1e-8);
  }
}

void DataUtils::denormalize(std::vector<double> &data,
                            const std::vector<double> &minVals,
                            const std::vector<double> &maxVals) {
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = data[i] * (maxVals[i] - minVals[i]) + minVals[i];
  }
}
