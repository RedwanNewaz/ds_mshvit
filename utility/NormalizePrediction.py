import pandas as pd 
import numpy as np 
import sys 
from pathlib import Path
import os 
# PRED_FILE = "/home/aredwann/PyDev/SewerML-tracker/results/rgb_vit/dev/benchmark_prediction_val.csv"
# OUT_FILE = "/home/aredwann/PyDev/SewerML-tracker/results/rgb_vit/dev/normalized/benchmark_prediction_val_normalized.csv"
def normalize(data):
    # Normalize the array using Min-Max scaling
    min_value = np.min(data)
    max_value = np.mean(data) + np.std(data)

    normalized_array = (data - min_value) / (max_value - min_value)
    return normalized_array


if __name__ == "__main__":
    root = sys.argv[1]
    for PRED_FILE in Path(root).glob('*.csv'):
        data = pd.read_csv(PRED_FILE)

        print(data.columns)
      
        for column in data.columns[1:]:
            data[column] = normalize(data[column])
        OUT_FILE = os.path.join(PRED_FILE.parent, 'normalized', PRED_FILE.name)
        data.to_csv(OUT_FILE)
      