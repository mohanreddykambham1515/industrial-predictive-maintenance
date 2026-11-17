from src.data import load_dataset
from src.preprocessing import build_preprocessor
from src.config import FEATURES

df = load_dataset()

X = df[FEATURES]

preprocessor = build_preprocessor()

X_transformed = preprocessor.fit_transform(X)

print("Original shape:", X.shape)
print("Transformed shape:", X_transformed.shape)