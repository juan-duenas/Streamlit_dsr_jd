import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Data source
DATA_URL = "https://raw.githubusercontent.com/JohannaViktor/streamlit_practical/refs/heads/main/global_development_data.csv"

# Features and target
FEATURES = ['GDP per capita', 'headcount_ratio_upper_mid_income_povline', 'year']
TARGET = 'Life Expectancy (IHME)'

# Load data
df = pd.read_csv(DATA_URL)

# Drop rows with missing values in features or target
df = df.dropna(subset=FEATURES + [TARGET])

X = df[FEATURES]
y = df[TARGET]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Save model to pickle file
import pickle
with open("rf_lifeexp_model.pkl", "wb") as f:
	pickle.dump(model, f)

# Predict and evaluate
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)
print(f"RandomForestRegressor R^2 score: {score:.3f}")
