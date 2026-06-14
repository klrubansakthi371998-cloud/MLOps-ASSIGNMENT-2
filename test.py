import joblib
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

# Retrieve the dataset to recreate the exact test environment
dataset = fetch_olivetti_faces()
features = dataset.data
labels = dataset.target

# Maintain random_state=42 to guarantee we test on the correct 30% split
_, X_eval, _, y_eval = train_test_split(features, labels, test_size=0.3, random_state=42)

print("Initializing evaluation phase...")
print("Loading 'savedmodel.pth' into memory...")
clf = joblib.load('savedmodel.pth')

# Calculate the final accuracy score
model_accuracy = clf.score(X_eval, y_eval)

print("----------------------------------------")
print(f"Evaluation Successful | Accuracy Score: {model_accuracy * 100:.4f}%")
print("----------------------------------------")