// NeuralNetwork.h
#pragma once

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

enum class ActivationFunction { RELU, SIGMOID, TANH, LEAKY_RELU };

class NeuralNetwork {
public:
  NeuralNetwork(int inputSize, int hiddenSize, int outputSize);

  void initializeWeights();
  void forwardPass(const std::vector<double> &input,
                   std::vector<double> &output);
  void backwardPass(const std::vector<double> &input,
                    const std::vector<double> &target, double learningRate);
  void train(const std::vector<std::vector<double>> &inputs,
             const std::vector<std::vector<double>> &targets, int epochs,
             double learningRate, ActivationFunction activationFunction);
  void saveModel(const std::string &filename) const;
  void loadModel(const std::string &filename);

  void setActivationFunction(ActivationFunction func);

private:
  int inputSize;
  int hiddenSize;
  int outputSize;

  std::vector<std::vector<double>> weightsInputHidden;
  std::vector<std::vector<double>> weightsHiddenOutput;
  std::vector<double> biasesHidden;
  std::vector<double> biasesOutput;
  std::vector<double> hiddenOutputs;

  ActivationFunction activationFunction;

  double activate(double x);
  double activateDerivative(double x);
};
