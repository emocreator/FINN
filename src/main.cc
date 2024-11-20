#include "../include/lossfunction.h"
#include "../include/neuralnetwork.h"
#include "../include/utils.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

void loadData(std::string_view filename,
              std::vector<std::vector<double>> &inputs,
              std::vector<double> &targets);

int main() {
  // Load data
  std::vector<std::vector<double>> inputs;
  std::vector<double> targets;
  loadData("financial_data.csv", inputs, targets);

  // Normalize data
  DataUtils::normalize(inputs);
  DataUtils::normalize(targets);

  // Split data into training and testing sets
  size_t totalSamples = inputs.size();
  size_t trainSamples = static_cast<size_t>(totalSamples * 0.8);

  std::vector<std::vector<double>> trainInputs(inputs.begin(),
                                               inputs.begin() + trainSamples);
  std::vector<double> trainTargets(targets.begin(),
                                   targets.begin() + trainSamples);

  std::vector<std::vector<double>> testInputs(inputs.begin() + trainSamples,
                                              inputs.end());
  std::vector<double> testTargets(targets.begin() + trainSamples,
                                  targets.end());

  // Create neural network
  int inputSize = static_cast<int>(trainInputs[0].size());
  int hiddenSize = 10;
  int outputSize = 1;

  NeuralNetwork nn(inputSize, hiddenSize, outputSize);

  // Set activation function
  nn.setActivationFunction(ActivationFunction::RELU);

  // Convert trainTargets to vector of vectors for compatibility
  std::vector<std::vector<double>> trainTargetsVec(trainTargets.size(),
                                                   std::vector<double>(1));
  for (size_t i = 0; i < trainTargets.size(); ++i) {
    trainTargetsVec[i][0] = trainTargets[i];
  }

  // Train neural network
  int epochs = 1000;
  double learningRate = 0.001;
  nn.train(trainInputs, trainTargetsVec, epochs, learningRate,
           ActivationFunction::RELU);

  // Evaluate on test data
  FinanceLossFunctions lossFunc;
  std::vector<double> predictedPrices;
  for (const auto &input : testInputs) {
    std::vector<double> output;
    nn.forwardPass(input, output);
    predictedPrices.push_back(output[0]);
  }

  double loss = lossFunc.computeLoss(predictedPrices, testTargets, testInputs);
  std::cout << "Test Loss (including arbitrage penalty): " << loss << '\n';

  // Save model
  nn.saveModel("finance_model.txt");

  return 0;
}

void loadData(std::string_view filename,
              std::vector<std::vector<double>> &inputs,
              std::vector<double> &targets) {
  std::ifstream file{std::string(filename)};
  if (!file) {
    throw std::runtime_error("Failed to open file: " + std::string(filename));
  }

  std::string line;
  size_t lineNum = 0;

  while (std::getline(file, line)) {
    lineNum++;
    if (line.empty())
      continue;

    std::vector<double> inputRow;
    std::stringstream ss(line);
    std::string value;
    bool first_column = true;

    while (std::getline(ss, value, ',')) {
      value.erase(0, value.find_first_not_of(" \t\r\n"));
      value.erase(value.find_last_not_of(" \t\r\n") + 1);

      if (first_column) {
        first_column = false;
        continue; // Skip date column
      }

      try {
        double num = std::stod(value);
        inputRow.push_back(num);
      } catch (const std::exception &e) {
        std::cerr << "Line " << lineNum << ": " << line << std::endl;
        throw std::runtime_error("Invalid numeric value at line " +
                                 std::to_string(lineNum) + ": " + value);
      }
    }

    if (inputRow.size() < 2) {
      throw std::runtime_error("Insufficient columns at line " +
                               std::to_string(lineNum));
    }

    targets.push_back(inputRow.back());
    inputRow.pop_back();
    inputs.push_back(std::move(inputRow));
  }

  if (inputs.empty()) {
    throw std::runtime_error("No valid data found in file");
  }
}
