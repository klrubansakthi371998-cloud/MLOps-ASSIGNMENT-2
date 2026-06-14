import joblib
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

# Reload data to fetch the exact same test subset splits
data = fetch_olivetti_faces()
X, y = data.data, data.target
_, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("📂 Loading savedmodel.pth...")
model = joblib.load('savedmodel.pth')

# Compute score
accuracy = model.score(X_test, y_test)
print("\n==============================")
print(f"📊 Final Test Accuracy: {accuracy * 100:.2f}%")
print("==============================")