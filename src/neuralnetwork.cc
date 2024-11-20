// NeuralNetwork.cc
#include "../include/neuralnetwork.h"

NeuralNetwork::NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
    : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize) {
  initializeWeights();
  activationFunction = ActivationFunction::RELU; // Default activation function
}

void NeuralNetwork::initializeWeights() {
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);

  weightsInputHidden.resize(inputSize, std::vector<double>(hiddenSize));
  for (auto &row : weightsInputHidden)
    for (auto &val : row)
      val = distribution(generator);

  weightsHiddenOutput.resize(hiddenSize, std::vector<double>(outputSize));
  for (auto &row : weightsHiddenOutput)
    for (auto &val : row)
      val = distribution(generator);

  biasesHidden.resize(hiddenSize);
  for (auto &val : biasesHidden)
    val = distribution(generator);

  biasesOutput.resize(outputSize);
  for (auto &val : biasesOutput)
    val = distribution(generator);
}

double NeuralNetwork::activate(double x) {
  switch (activationFunction) {
  case ActivationFunction::RELU:
    return std::max(0.0, x);
  case ActivationFunction::SIGMOID:
    return 1.0 / (1.0 + std::exp(-x));
  case ActivationFunction::TANH:
    return std::tanh(x);
  case ActivationFunction::LEAKY_RELU:
    return x > 0 ? x : 0.01 * x;
  default:
    return x;
  }
}

double NeuralNetwork::activateDerivative(double x) {
  switch (activationFunction) {
  case ActivationFunction::RELU:
    return x > 0 ? 1.0 : 0.0;
  case ActivationFunction::SIGMOID: {
    double sigmoid = 1.0 / (1.0 + std::exp(-x));
    return sigmoid * (1 - sigmoid);
  }
  case ActivationFunction::TANH:
    return 1.0 - std::tanh(x) * std::tanh(x);
  case ActivationFunction::LEAKY_RELU:
    return x > 0 ? 1.0 : 0.01;
  default:
    return 1.0;
  }
}

void NeuralNetwork::forwardPass(const std::vector<double> &input,
                                std::vector<double> &output) {
  // Compute hidden layer activations
  hiddenOutputs.resize(hiddenSize);
  for (int j = 0; j < hiddenSize; ++j) {
    double sum = biasesHidden[j];
    for (int i = 0; i < inputSize; ++i) {
      sum += input[i] * weightsInputHidden[i][j];
    }
    hiddenOutputs[j] = activate(sum);
  }

  // Compute output layer activations
  output.resize(outputSize);
  for (int k = 0; k < outputSize; ++k) {
    double sum = biasesOutput[k];
    for (int j = 0; j < hiddenSize; ++j) {
      sum += hiddenOutputs[j] * weightsHiddenOutput[j][k];
    }
    output[k] = sum; // Linear activation for output layer
  }
}

void NeuralNetwork::backwardPass(const std::vector<double> &input,
                                 const std::vector<double> &target,
                                 double learningRate) {
  // Forward pass to get outputs
  std::vector<double> outputs;
  forwardPass(input, outputs);

  // Compute output errors (assuming linear activation on output layer)
  std::vector<double> outputErrors(outputSize);
  for (int k = 0; k < outputSize; ++k) {
    outputErrors[k] = target[k] - outputs[k];
  }

  // Compute hidden errors
  std::vector<double> hiddenErrors(hiddenSize);
  for (int j = 0; j < hiddenSize; ++j) {
    double error = 0.0;
    for (int k = 0; k < outputSize; ++k) {
      error += outputErrors[k] * weightsHiddenOutput[j][k];
    }
    hiddenErrors[j] = error * activateDerivative(hiddenOutputs[j]);
  }

  // Update weights and biases between hidden and output layers
  for (int j = 0; j < hiddenSize; ++j) {
    for (int k = 0; k < outputSize; ++k) {
      weightsHiddenOutput[j][k] +=
          learningRate * outputErrors[k] * hiddenOutputs[j];
    }
  }

  for (int k = 0; k < outputSize; ++k) {
    biasesOutput[k] += learningRate * outputErrors[k];
  }

  // Update weights and biases between input and hidden layers
  for (int i = 0; i < inputSize; ++i) {
    for (int j = 0; j < hiddenSize; ++j) {
      weightsInputHidden[i][j] += learningRate * hiddenErrors[j] * input[i];
    }
  }

  for (int j = 0; j < hiddenSize; ++j) {
    biasesHidden[j] += learningRate * hiddenErrors[j];
  }
}

void NeuralNetwork::train(const std::vector<std::vector<double>> &inputs,
                          const std::vector<std::vector<double>> &targets,
                          int epochs, double learningRate,
                          ActivationFunction activationFunction) {
  this->activationFunction = activationFunction;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    double totalLoss = 0.0;
    for (size_t i = 0; i < inputs.size(); ++i) {
      backwardPass(inputs[i], targets[i], learningRate);

      // Compute loss (Mean Squared Error)
      std::vector<double> outputs;
      forwardPass(inputs[i], outputs);
      for (size_t k = 0; k < outputs.size(); ++k) {
        double error = targets[i][k] - outputs[k];
        totalLoss += error * error;
      }
    }
    totalLoss /= inputs.size();
    if (epoch % 100 == 0) {
      std::cout << "Epoch " << epoch << ", Loss: " << totalLoss << std::endl;
    }
  }
}

void NeuralNetwork::saveModel(const std::string &filename) const {
  std::ofstream outFile(filename);
  if (!outFile) {
    std::cerr << "Error opening file for saving model." << std::endl;
    return;
  }

  // Save weights and biases
  for (const auto &row : weightsInputHidden)
    for (const auto &val : row)
      outFile << val << " ";
  outFile << std::endl;

  for (const auto &row : weightsHiddenOutput)
    for (const auto &val : row)
      outFile << val << " ";
  outFile << std::endl;

  for (const auto &val : biasesHidden)
    outFile << val << " ";
  outFile << std::endl;

  for (const auto &val : biasesOutput)
    outFile << val << " ";
  outFile << std::endl;

  outFile.close();
}

void NeuralNetwork::loadModel(const std::string &filename) {
  std::ifstream inFile(filename);
  if (!inFile) {
    std::cerr << "Error opening file for loading model." << std::endl;
    return;
  }

  // Load weights and biases
  for (auto &row : weightsInputHidden)
    for (auto &val : row)
      inFile >> val;

  for (auto &row : weightsHiddenOutput)
    for (auto &val : row)
      inFile >> val;

  for (auto &val : biasesHidden)
    inFile >> val;

  for (auto &val : biasesOutput)
    inFile >> val;

  inFile.close();
}

void NeuralNetwork::setActivationFunction(ActivationFunction func) {
  activationFunction = func;
}
