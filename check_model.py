import os
import pickle

# ✅ Logistic Regression ki pkl
pkl_path = r"D:\project\easieast_way\best_model_Logistic_Regression.pkl"

# File size dekho
size = os.path.getsize(pkl_path)
print(f"File size: {size} bytes")

# Load karke dekho
with open(pkl_path, "rb") as f:
    model = pickle.load(f)

print(f"Model type: {type(model)}")
print(model)