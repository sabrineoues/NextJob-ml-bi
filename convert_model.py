from tensorflow.keras.models import load_model

old_model_path = "ml_models/encoder_model.h5"
model = load_model(old_model_path, compile=False)

new_model_path = "ml_models/encoder_model.keras"
model.save(new_model_path)

print(f"Modèle converti en {new_model_path}")
