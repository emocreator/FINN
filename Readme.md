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

## Results

```
Epoch 0, Loss: 0.0359018
Epoch 100, Loss: 0.000634882
Epoch 200, Loss: 0.000337825
Epoch 300, Loss: 0.000195165
Epoch 400, Loss: 0.000118311
Epoch 500, Loss: 7.44451e-05
Epoch 600, Loss: 4.85446e-05
Epoch 700, Loss: 3.28287e-05
Epoch 800, Loss: 2.3069e-05
Epoch 900, Loss: 1.68505e-05
Test Loss (including arbitrage penalty): 0.00486601
```

## Dependencies

- C++20 compiler
- Make

## License

MIT License

## Acknowledgments

Core concept inspired by tsotchke's [PINN project](https://github.com/tsotchke/PINN), adapting physics-informed principles to financial modeling.
