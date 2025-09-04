# Movie Rating Prediction Model (Auto-Detect Columns)

# 1. Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2. Load dataset
file_path = r"E:/CODSOFT/IMDb Movies India.csv"  # Adjust path if needed
df = pd.read_csv(file_path, encoding='latin1')

# 3. Show dataset info
print("Columns in dataset:", df.columns.tolist())
print(df.head())

# 4. Find Rating column (case-insensitive)
rating_col = None
for col in df.columns:
    if 'rating' in col.lower():
        rating_col = col
        break

if rating_col is None:
    raise ValueError("No 'Rating' column found. Please check your CSV file!")

# 5. Drop rows with missing ratings
df = df.dropna(subset=[rating_col])

# 6. Fill missing values in other columns
df = df.fillna('Unknown')

# 7. Select features and target
target = rating_col
# Use all object (categorical) columns except target
categorical_cols = df.select_dtypes(include='object').columns.tolist()
if target in categorical_cols:
    categorical_cols.remove(target)

X = df[categorical_cols]
y = df[target]

print(f"Using features: {categorical_cols}")
print(f"Target: {target}")

# 8. Preprocessing - OneHotEncode categorical columns
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)]
)

# 9. Define model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 10. Create pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# 11. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 12. Train model
pipeline.fit(X_train, y_train)

# 13. Make predictions
y_pred = pipeline.predict(X_test)

# 14. Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# 15. Sample prediction
sample_data = {col: [X[col].iloc[0]] for col in categorical_cols}  # Take first row's data
sample_df = pd.DataFrame(sample_data)
sample_pred = pipeline.predict(sample_df)
print(f"\nSample Movie Predicted Rating: {sample_pred[0]:.2f}")
