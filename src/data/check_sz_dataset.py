"""
Quickly verifies SZ-Taxi dataset files and prints basic info.
"""
import pandas as pd

def preview_sz_data():
    adj = pd.read_csv("data/sz_adj.csv", header=None)
    speed = pd.read_csv("data/sz_speed.csv")
    print("Adjacency shape:", adj.shape)
    print("Speed data shape:", speed.shape)
    print("Speed data sample:")
    print(speed.head())

if __name__ == "__main__":
    preview_sz_data()
