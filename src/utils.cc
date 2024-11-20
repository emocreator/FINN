#include "../include/utils.h"
#include <stdexcept>

void DataUtils::normalize(std::vector<std::vector<double>> &data) {
  if (data.empty())
    return;

  const size_t numFeatures = data[0].size();
  const size_t numSamples = data.size();

  std::vector<double> minVals(numFeatures, std::numeric_limits<double>::max());
  std::vector<double> maxVals(numFeatures,
                              std::numeric_limits<double>::lowest());

  for (size_t i = 0; i < numSamples; ++i) {
    if (data[i].size() != numFeatures) {
      throw std::runtime_error("Row " + std::to_string(i) +
                               " has inconsistent features");
    }
    for (size_t j = 0; j < numFeatures; ++j) {
      minVals[j] = std::min(minVals[j], data[i][j]);
      maxVals[j] = std::max(maxVals[j], data[i][j]);
    }
  }

  for (size_t i = 0; i < numSamples; ++i) {
    for (size_t j = 0; j < numFeatures; ++j) {
      double range = maxVals[j] - minVals[j];
      if (range > 1e-10) {
        data[i][j] = (data[i][j] - minVals[j]) / range;
      } else {
        data[i][j] = 0.0;
      }
    }
  }
}

void DataUtils::normalize(std::vector<double> &data) {
  if (data.empty())
    return;

  double minVal = *std::min_element(data.begin(), data.end());
  double maxVal = *std::max_element(data.begin(), data.end());
  double range = maxVal - minVal + 1e-8;

  for (auto &val : data) {
    val = (val - minVal) / range;
  }
}

void DataUtils::denormalize(std::vector<double> &data,
                            const std::vector<double> &minVals,
                            const std::vector<double> &maxVals) {
  for (size_t i = 0; i < data.size(); ++i) {
    double range = maxVals[i] - minVals[i];
    data[i] = data[i] * range + minVals[i];
  }
}
