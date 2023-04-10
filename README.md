# Stock Market price Prediction

![model](https://img.shields.io/badge/Task-Forcasting-green) ![coding](https://img.shields.io/badge/Model-LSTM-yellow) 

This project uses LSTM (Long Short-Term Memory) neural networks to predict the stock prices of a given company based on historical data. LSTM is a type of Recurrent Neural Network (RNN) that can effectively model sequences of data, making it a good choice for time-series analysis and prediction. In this project, we use historical stock prices as input to train an LSTM model, and then use the trained model to predict future stock prices. We evaluate the performance of the model by comparing the predicted prices with the actual prices.

# Customization
You can customize this project by modifying the following parameters in the config.py file:

 - look_back: the number of past days to use as input for the LSTM model  
 - epochs: the number of epochs to train the model  
 - batch_size: the batch size for training the model  
 - train_test_split: the ratio of data to use for training vs testing the model  

You can also modify the LSTM architecture in the main.py file by changing the number of LSTM layers, the number of neurons in each layer, and the activation function used.



