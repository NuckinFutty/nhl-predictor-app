import pandas as pd
import numpy as np
import sqlite3
from scipy.stats import zscore

# Load NHL game data from the database
def load_data(db_path="nhl_game_data.db"):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM game_results", conn)
    conn.close()
    return df

# Function to detect outliers using Z-score
def detect_outliers(df, column, threshold=2.5):
    df['z_score'] = zscore(df[column])
    outliers = df[np.abs(df['z_score']) > threshold]
    return outliers

# Example usage
if __name__ == "__main__":
    df = load_data()
    outliers = detect_outliers(df, 'home_score')
    print("Outlier Games:", outliers)
