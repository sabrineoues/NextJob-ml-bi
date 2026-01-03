import os
import os
import joblib
import tensorflow as tf
import numpy as np
import os
# Supprime les warnings info/warning de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 



def _load_model(filename):
    """Try to load a model using joblib first, then pickle as a fallback.

    Raises FileNotFoundError if the file does not exist, and re-raises
    the underlying loader exception if both attempts fail.
    """
    path = os.path.join(os.path.dirname(__file__), 'ml_models', filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    # First try joblib (commonly used for scikit-learn pipelines)
    try:
        import joblib

        return joblib.load(path)
    except Exception as job_err:
        # Fallback to pickle
        try:
            import pickle

            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as p_err:
            # Provide both errors to help debugging
            raise RuntimeError(
                f"Failed to load model '{path}': joblib error: {job_err!r}; pickle error: {p_err!r}"
            ) from p_err




BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "ml_models")

def load_skills_gap_models():
    """
    Charge le modèle Keras (.h5) et les objets sklearn (.pkl)
    Retourne un dictionnaire contenant:
    encoder, classifier, le, skills_cols, top_skills_by_title
    """
    # 1️⃣ Charger le modèle Keras (.h5)
    encoder_path = r"C:\3eme\power1\ml_platform\ml_app\ml_models\encoder_model.h5"
    encoder_model = tf.keras.models.load_model(encoder_path, compile=False)

    # 2️⃣ Charger les objets sklearn
    data_path = os.path.join(MODEL_DIR, "final_small_classifier2.pkl")
    data = joblib.load(data_path)

    return {
        "encoder": encoder_model,
        "classifier": data["classifier"],
        "le": data["le"],
        "skills_cols": data["skills_cols"],
        "top_skills_by_title": data["top_skills_by_title"]
    }
def load_encoder_model():
    return load_skills_gap_models()["encoder"]