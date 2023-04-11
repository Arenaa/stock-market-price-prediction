import configparser

import torch
import tqdm

from data_setup import create_dataloaders
from src.model import LSTM
from src.train import train

config  = configparser.ConfigParser()

config.read('src/config/config.ini')

lookback  = config.getint('model', 'lookback')
train_test_split = config.getfloat('model', 'train_test_split')
batch_size =  config.getint( 'model', 'batch_size')
input_size =  config.getint( 'model', 'input_size')
hidden_size =  config.getint( 'model', 'hidden_size')
num_stacked_layers =  config.getint( 'model', 'num_stacked_layers')
epochs = config.getint('model', 'epochs')
data_dir = config.get('model', 'data_dir')

device = "cuda" if torch.cuda.is_available() else "cpu"

model = LSTM(input_size, hidden_size, num_stacked_layers, device)
train_dataloader, test_dataloader = create_dataloaders( data_dir=data_dir,
                                                        lookback=lookback,
                                                         batch_size=batch_size,
                                                          train_test_split=train_test_split )

if __name__==  "__main__":
    train(model,
          train_dataloader,
          test_dataloader,
          epochs,
          device)
