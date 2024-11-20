#pragma once
#include <algorithm>
#include <limits>
#include <vector>

class DataUtils {
public:
  static void normalize(std::vector<std::vector<double>> &data);
  static void normalize(std::vector<double> &data);
  static void denormalize(std::vector<double> &data,
                          const std::vector<double> &minVals,
                          const std::vector<double> &maxVals);
};
