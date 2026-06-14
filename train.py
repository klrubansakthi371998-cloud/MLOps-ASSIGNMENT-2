import joblib
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

print("🔄 Automatically fetching the Olivetti faces dataset...")
data = fetch_olivetti_faces()
X, y = data.data, data.target

print("✂️ Splitting the data: 70% Train and 30% Test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("🧠 Training scikit-learn DecisionTreeClassifier...")
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

print("💾 Saving the trained model file...")
joblib.dump(model, 'savedmodel.pth')
print("✅ Success! 'savedmodel.pth' generated.")