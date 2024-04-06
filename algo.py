import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from agent import *
from nets import *

sns.set()

def simulate_trades(initial_money=10000, window_size=30, skip=1, population_size=100, generations=100, mutation_rate=0.1):
    # Fetch data from Yahoo Finance
    msft = yf.Ticker("MSFT")
    data = msft.history(start="2023-01-01", end="2024-02-19")
    
    # Split the data into training (January to June) and testing (June to present)
    train_data = data.loc["2023-01-01":"2023-06-30"]
    test_data = data.loc["2023-06-01":]
    
    # Train the initial model using data from January to June
    train_df = train_data.copy()
    count = int(np.ceil(len(train_df) * 0.1))
    signals = pd.DataFrame(index=train_df.index)
    signals['signal'] = 0.0
    signals['trend'] = train_df['Close']
    signals['RollingMax'] = (signals.trend.shift(1).rolling(count).max())
    signals['RollingMin'] = (signals.trend.shift(1).rolling(count).min())
    signals.loc[signals['RollingMax'] < signals.trend, 'signal'] = -1
    signals.loc[signals['RollingMin'] > signals.trend, 'signal'] = 1
    close = train_df['Close']
    
    neural_evolve = NeuroEvolution(population_size, mutation_rate, neuralnetwork,
                                  window_size, window_size, close, skip, initial_money)
    fittest_nets = neural_evolve.evolve(generations)
    
    # Simulate trades using data from June to present
    test_df = test_data.copy()
    signals = pd.DataFrame(index=test_df.index)
    signals['signal'] = 0.0
    signals['trend'] = test_df['Close']
    signals['RollingMax'] = (signals.trend.shift(1).rolling(count).max())
    signals['RollingMin'] = (signals.trend.shift(1).rolling(count).min())
    signals.loc[signals['RollingMax'] < signals.trend, 'signal'] = -1
    signals.loc[signals['RollingMin'] > signals.trend, 'signal'] = 1
    close = test_df['Close']
    
    states_buy, states_sell, total_gains, invest = neural_evolve.buy(fittest_nets)
    
    # Update the model with the new data from June to present
    train_df = pd.concat([train_df, test_df])
    count = int(np.ceil(len(train_df) * 0.1))
    signals = pd.DataFrame(index=train_df.index)
    signals['signal'] = 0.0
    signals['trend'] = train_df['Close']
    signals['RollingMax'] = (signals.trend.shift(1).rolling(count).max())
    signals['RollingMin'] = (signals.trend.shift(1).rolling(count).min())
    signals.loc[signals['RollingMax'] < signals.trend, 'signal'] = -1
    signals.loc[signals['RollingMin'] > signals.trend, 'signal'] = 1
    close = train_df['Close']
    
    neural_evolve = NeuroEvolution(population_size, mutation_rate, neuralnetwork,
                                  window_size, window_size, close, skip, initial_money)
    fittest_nets = neural_evolve.evolve(generations)
    
    return states_buy, states_sell, total_gains, invest

states_buy, states_sell, total_gains, invest = simulate_trades()
