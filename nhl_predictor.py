import streamlit as st
import sqlite3
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# Load team stats from database
conn = sqlite3.connect("nhl_data.db")
df_stats = pd.read_sql("SELECT * FROM team_stats", conn)
conn.close()

import os

st.write("Current Directory:", os.getcwd())
st.write("Files in Directory:", os.listdir())

# Check if the file exists
st.write("Does nhl_model.pkl exist?", os.path.isfile("nhl_model.pkl"))


# Load trained model & scaler
with open("nhl_model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

# Streamlit UI
st.title("🏒 NHL Game Predictor")
st.write("Select two teams to predict the winner!")

# Team selection dropdowns
teams = df_stats["team"].tolist()
home_team = st.selectbox("Select Home Team", teams)
away_team = st.selectbox("Select Away Team", teams)

# Predict button
if st.button("Predict Outcome"):
    if home_team == away_team:
        st.error("Home and Away teams must be different!")
    else:
        # Get team stats
        home_stats = df_stats[df_stats["team"] == home_team].drop(columns=["team"])
        away_stats = df_stats[df_stats["team"] == away_team].drop(columns=["team"])
        
        if home_stats.empty or away_stats.empty:
            st.error("Team stats not available!")
        else:
            # Merge stats
            game_features = pd.concat([home_stats.reset_index(drop=True), away_stats.reset_index(drop=True)], axis=1)
            
            # Standardize input
            game_features = scaler.transform(game_features)
            
            # Predict outcome
            prediction = model.predict_proba(game_features)[0][1]  # Probability of home team winning
            
            # Display result
            st.success(f"🏒 {home_team} has a {prediction*100:.2f}% chance of winning!")
