import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv("C:\\Projects\\clf\\insurance.csv")

X = df.drop(columns=["charges"])
y = df["charges"]

numerical_cols = ["age", "bmi", "children"]
categorical_cols = ["sex", "smoker", "region"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(drop="first"), categorical_cols)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

X_transformed = preprocessor.fit_transform(X)

print("Transformed shape:", X_transformed.shape)

y_pred = model.predict(X_test)

print("Predictions:", y_pred)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R^2 Score:", r2)

import joblib
joblib.dump(model, "insurance_model.pkl")


