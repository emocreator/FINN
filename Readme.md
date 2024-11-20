# FINN (Finance-Informed Neural Network)

A neural network that incorporates financial principles and constraints into its learning process, inspired by [Physics-Informed Neural Networks (PINN)](https://github.com/tsotchke/PINN).

## Features

- Custom loss functions incorporating financial constraints and arbitrage penalties
- Multiple activation functions (ReLU, Sigmoid, Tanh, Leaky ReLU)
- Data normalization and preprocessing utilities
- Model save/load functionality

## Usage

```bash
make
cd ./bin/
./finn
```

## Dependencies

- C++20 compiler
- Make

## License

MIT License

## Acknowledgments

Core concept inspired by tsotchke's [PINN project](https://github.com/tsotchke/PINN), adapting physics-informed principles to financial modeling.
