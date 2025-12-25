from django.shortcuts import render
from .forms import KNNSampleForm, RegressionForm
from django.conf import settings
import joblib
from .forms import XGBForm
import os
import logging
import numpy as np
import pandas as pd
from .ml_utils import _load_model



logger = logging.getLogger(__name__)
# Charger le dataset pour calculer les fréquences automatiquement
df = pd.read_csv('C:\\3eme\\datawarehouse\\dataset_ceaned.csv')  # ou depuis la base Django
df1 = pd.read_csv('C:\\3eme\\datawarehouse\\DataScience_salaries_2025.csv')  # ou depuis la base Django
try:
    df = pd.read_csv('C:\\3eme\\datawarehouse\\dataset_ceaned.csv')  # ou depuis la base Django
except Exception as e:
    logger.warning("Could not load dataset_ceaned.csv: %s", e)
    df = pd.DataFrame()

try:
    df1 = pd.read_csv('C:\\3eme\\datawarehouse\\DataScience_salaries_2025.csv')  # ou depuis la base Django
except Exception as e:
    logger.warning("Could not load DataScience_salaries_2025.csv: %s", e)
    df1 = pd.DataFrame()
def knn_view(request):
    form = KNNSampleForm(request.POST or None)
    result = None
    label = None
    confidence = None
    model_missing = False

    try:
        knn_pipeline = _load_model('knn.pkl')
    except FileNotFoundError:
        knn_pipeline = None
        model_missing = True

    if form.is_valid() and knn_pipeline is not None:
        try:
            company_name = form.cleaned_data['company_name']
            location = form.cleaned_data['location']
            salary = form.cleaned_data['salary']

            # Calcul des fréquences automatiquement
            company_name_freq = df['company_name'].value_counts(normalize=True).get(company_name, 0.0001)
            location_freq = df['Location'].value_counts(normalize=True).get(location, 0.0001)

            # Préparer les données dans le bon ordre attendu par le pipeline
            X = np.array([[salary, company_name_freq, location_freq]])

            # Prédiction
            result = knn_pipeline.predict(X)[0]

            # Mapping label
            joblevel_label = {0: 'Junior', 1: 'Medior', 2: 'Senior'}
            label = joblevel_label.get(int(result), str(result))

            # Confiance (%)
            if hasattr(knn_pipeline, 'predict_proba'):
                probs = knn_pipeline.predict_proba(X)[0]
                confidence = round(float(np.max(probs) * 100), 2)

        except Exception:
            logger.exception("Erreur lors de la prédiction KNN")
            result, label, confidence = None, None, None

    return render(request, 'knn.html', {
        'form': form,
        'result': result,
        'label': label,
        'confidence': confidence,
        'model_missing': model_missing,
    })

def xgb_view(request):
    form = XGBForm(request.POST or None)
    result = None
    label = None
    confidence = None
    model_missing = False

    # Charger le modèle XGBoost
    try:
        xgb_model = _load_model('xgb_model.pkl')
    except FileNotFoundError:
        xgb_model = None
        model_missing = True

    if form.is_valid() and xgb_model is not None:
        try:
            # Récupérer les entrées utilisateur
            location = form.cleaned_data['location']
            skill = form.cleaned_data['skill']
            company_name = form.cleaned_data['company_name']
            platform_name = form.cleaned_data['platform_name']
            degree = form.cleaned_data['degree']

            # Transformation des mots en fréquences / scores
            location_freq = df['Location'].value_counts(normalize=True).get(location, 0)
            skill_score = df['skill_score'].value_counts(normalize=True).get(skill, 0)
            company_freq = df['company_name'].value_counts(normalize=True).get(company_name, 0)
            platform_freq = df['platform_name'].value_counts(normalize=True).get(platform_name, 0)
            # Pour degree, encode avec factorize
            degree_encoded = pd.factorize(df['degree'])[1].get_loc(degree) if degree in df['degree'].values else 0

            # Préparer les données dans l'ordre exact attendu par le modèle
            X = np.array([[location_freq, skill_score, company_freq, platform_freq, degree_encoded]])

            # Prédiction
            result = int(xgb_model.predict(X)[0])

            # Mapping label binaire
            insurance_label = {0: "No Insurance", 1: "Insurance"}
            label = insurance_label.get(result, "Unknown")

            # Confiance (%)
            if hasattr(xgb_model, 'predict_proba'):
                probs = xgb_model.predict_proba(X)[0]
                confidence = round(float(np.max(probs) * 100), 2)

        except Exception:
            logger.exception("Erreur lors de la prédiction XGBoost")
            result, label, confidence = None, None, None

    return render(request, 'xgboost.html', {
        'form': form,
        'result': result,
        'label': label,
        'confidence': confidence,
        'model_missing': model_missing,
    })

def home(request):
    return render(request, 'home.html')


def dashboard(request):
    return render(request, 'dashboard.html')




def _load_model(filename):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # dossier courant de l'app
    model_path = os.path.join(BASE_DIR, "ml_models", filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")
    return joblib.load(model_path)

# Charger modèle XGBoost et features (robuste si fichiers manquants)
salary_model = None
feature_order = []

def _load_salary_model():
    """Charge le modèle de salaire en essayant les variantes disponibles."""
    candidates = ["xgb_salary_model_opt.pkl", "xgb_salary_model.pkl"]
    last_err = None
    for name in candidates:
        try:
            return _load_model(name)
        except FileNotFoundError as exc:  # garder trace et continuer
            last_err = exc
    if last_err:
        logger.warning("XGB salary model introuvable: %s", last_err)
    return None


salary_model = _load_salary_model()

# Prioriser les features embarquees dans le modele; sinon fallback fichier
if salary_model is not None and hasattr(salary_model, "feature_names_in_"):
    feature_order = list(salary_model.feature_names_in_)
else:
    try:
        feature_order = _load_model("xgb_features_opt.pkl")
    except FileNotFoundError as exc:
        logger.warning("Fichier feature_order introuvable: %s", exc)
        feature_order = []


def _load_regression_dataset():
    """Charge le dataset salaire depuis les chemins connus."""
    candidates = [
        "C:\\3eme\\datawarehouse\\DataScience_salaries_2025.csv",
        os.path.join(os.path.dirname(__file__), "ml_models", "DataScience_salaries_2025.csv"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            return pd.read_csv(path)
        except Exception as exc:
            logger.warning("Echec de chargement du dataset %s: %s", path, exc)
    logger.warning("Aucun dataset salaire charge; encodage cible utilisera la moyenne par defaut")
    return pd.DataFrame()


df1 = _load_regression_dataset().drop_duplicates()

# Encodage ordinal pour df1
exp_map = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
company_size_map = {'S': 0, 'M': 1, 'L': 2}
employment_map = {'FT': 0, 'PT': 1, 'CT': 2, 'FL': 3}

if not df1.empty:
    if 'experience_level' in df1.columns:
        df1['experience_level'] = df1['experience_level'].map(exp_map)
    if 'company_size' in df1.columns:
        df1['company_size'] = df1['company_size'].map(company_size_map)
    if 'employment_type' in df1.columns:
        df1['employment_type'] = df1['employment_type'].map(employment_map)

DEFAULT_SALARY_MEAN = 60000.0


def _global_salary_mean():
    if not df1.empty and "salary_in_usd" in df1.columns:
        return float(df1["salary_in_usd"].mean())
    return DEFAULT_SALARY_MEAN

# -------------------------------
# Target encoding sécurisé
# -------------------------------
def get_target_encoding(col, val):
    """Retourne la moyenne du salaire pour val dans col ou la moyenne globale si inconnu"""
    if not df1.empty and col in df1.columns and "salary_in_usd" in df1.columns:
        te_map = df1.groupby(col)["salary_in_usd"].mean()
        return float(te_map.get(val, _global_salary_mean()))
    return _global_salary_mean()

# -------------------------------
# Vue Django pour la prédiction
# -------------------------------
def regression_view(request):
    form = RegressionForm(request.POST or None)
    result = None
    error_message = None
    model_missing = salary_model is None or not feature_order

    if model_missing:
        error_message = "Le modele de regression n'est pas disponible sur le serveur."

    if form.is_valid() and not model_missing:
        cd = form.cleaned_data
        try:
            # Construire un dictionnaire avec toutes les features
            row = {
                "work_year": float(cd["work_year"]),
                "experience_level": exp_map.get(cd["experience_level"], 0),
                "employment_type": employment_map.get(cd["employment_type"], 0),
                "remote_ratio": float(cd["remote_ratio"]),
                "company_size": company_size_map.get(cd["company_size"], 0),
                "job_title_te": get_target_encoding("job_title", cd["job_title"]),
                "company_location_te": get_target_encoding("company_location", cd["company_location"]),
                "employee_residence_te": get_target_encoding("employee_residence", cd["employee_residence"]),
                "is_remote": 1 if cd["remote_ratio"] == 100 else 0,
                "is_hybrid": 1 if 0 < cd["remote_ratio"] < 100 else 0
            }

            # Préparer X dans l'ordre exact des features
            try:
                X_input = np.array([[row[f] for f in feature_order]], dtype=float)
            except KeyError as exc:
                raise KeyError(f"Feature manquante pour le modele: {exc}")

            # Prédiction XGBoost + inverse log
            log_pred = salary_model.predict(X_input)[0]
            pred_salary = np.expm1(log_pred)
            result = round(float(pred_salary), 2)

            # Eviter les retours a zero en cas de pred invalide
            if result <= 0:
                result = round(_global_salary_mean(), 2)

        except Exception as e:
            error_message = str(e)

    return render(request, "regression.html", {
        "form": form,
        "result": result,
        "error_message": error_message,
        "model_missing": model_missing
    })

def kmeans_view(request):
    return render(request, 'kmeans.html')
