import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from feature_extraction import extract_features


print("STEP 1: Loading datasets")

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")


# Use smaller dataset for faster testing
train_data = train_data.head(3000)
test_data = test_data.head(1000)


print("STEP 2: Extracting features")

X_train, y_train = extract_features(train_data)
X_test, y_test = extract_features(test_data)


print("STEP 3: Training model")

model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)


print("STEP 4: Predicting")

pred = model.predict(X_test)


print("STEP 5: Calculating accuracy")

accuracy = accuracy_score(y_test, pred)

print("Model Accuracy:", accuracy)


print("STEP 6: Saving model")

joblib.dump(model, "models/eeg_model.pkl")

print("Model saved successfully")