import joblib
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

print("[INFO] Initiating Olivetti faces dataset extraction...")
dataset = fetch_olivetti_faces()
features = dataset.data
targets = dataset.target

print("[INFO] Creating 70/30 train-validation split...")
train_feat, test_feat, train_tgt, test_tgt = train_test_split(features, targets, test_size=0.3, random_state=42)

print("[INFO] Fitting Decision Tree algorithm to training features...")
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(train_feat, train_tgt)

print("[INFO] Serializing model to disk...")
joblib.dump(tree_clf, 'savedmodel.pth')

print("------------------------------------------------------------")
print("[SUCCESS] Training complete. Model exported as 'savedmodel.pth'.")
print("------------------------------------------------------------")