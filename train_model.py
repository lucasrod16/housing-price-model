import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv("Housing.csv")

X = data.drop(columns=["price"])
y = data["price"]

imputer = SimpleImputer(strategy="median")
X_numeric = X.select_dtypes(include=["number"])
X_imputed = imputer.fit_transform(X_numeric)

X_imputed_df = pd.DataFrame(X_imputed, columns=X_numeric.columns)

X_categorical = X.select_dtypes(include=["object"])
X_categorical_dummies = pd.get_dummies(X_categorical, drop_first=True)

X_processed = pd.concat([X_imputed_df, X_categorical_dummies], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared: {r2:.2f}")
