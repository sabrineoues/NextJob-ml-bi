# test.py
from ml_app.ml_utils import load_skills_gap_models

models = load_skills_gap_models()
print(models)
import numpy as np
import pandas as pd

if __name__ == "__main__":
    models = load_skills_gap_models()
    encoder = models["encoder"]
    classifier = models["classifier"]
    le = models["le"]
    skills_cols = models["skills_cols"]

    # Fake input pour tester
    data = np.random.rand(1, encoder.input_shape[1])
    encoded = encoder.predict(data)

    X_new = pd.DataFrame(encoded, columns=[f"f{i}" for i in range(encoded.shape[1])])
    pred = classifier.predict(X_new)
    print("Prediction:", le.inverse_transform(pred))
