import os
import logging
import numpy as np
import pandas as pd
from django.shortcuts import render
from .forms import KNNSampleForm, RegressionForm, XGBForm, RecommandationForm
from .ml_utils import _load_model, load_skills_gap_models, load_encoder_model
import tensorflow as tf
import random

logger = logging.getLogger(__name__)
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Charger les modèles Keras/sklearn
try:
    MODELS = load_skills_gap_models()
    encoder = MODELS["encoder"]
    classifier = MODELS["classifier"]
    le = MODELS["le"]
    skills_cols = MODELS["skills_cols"]
    top_skills_by_title = MODELS["top_skills_by_title"]
except Exception as e:
    logger.error("Impossible de charger les modèles: %s", e)
    encoder = classifier = le = skills_cols = top_skills_by_title = None

# Datasets
DATA_DIR = "C:\\3eme\\datawarehouse"  # adapter si besoin
try:
    df = pd.read_csv(os.path.join(DATA_DIR, "dataset_ceaned.csv"))
except Exception as e:
    logger.warning(f"Could not load dataset_ceaned.csv: {e}")
    df = pd.DataFrame()

try:
    df1 = pd.read_csv(os.path.join(DATA_DIR, "DataScience_salaries_2025.csv"))
except Exception as e:
    logger.warning(f"Could not load DataScience_salaries_2025.csv: {e}")
    df1 = pd.DataFrame()

# -------------------------------
# KNN VIEW
# -------------------------------
def knn_view(request):
    form = KNNSampleForm(request.POST or None)
    result = label = confidence = None
    model_missing = False

    try:
        knn_pipeline = _load_model("knn.pkl")
    except FileNotFoundError:
        knn_pipeline = None
        model_missing = True

    if form.is_valid() and knn_pipeline is not None:
        try:
            company_name = form.cleaned_data['company_name']
            location = form.cleaned_data['location']
            salary = form.cleaned_data['salary']

            company_freq = df['company_name'].value_counts(normalize=True).get(company_name, 0.0001)
            location_freq = df['Location'].value_counts(normalize=True).get(location, 0.0001)

            X = np.array([[salary, company_freq, location_freq]])

            result = knn_pipeline.predict(X)[0]
            joblevel_label = {0: 'Junior', 1: 'Medior', 2: 'Senior'}
            label = joblevel_label.get(int(result), str(result))

            if hasattr(knn_pipeline, 'predict_proba'):
                confidence = round(float(np.max(knn_pipeline.predict_proba(X)[0]) * 100), 2)

        except Exception:
            logger.exception("Erreur lors de la prédiction KNN")
            result = label = confidence = None

    return render(request, "knn.html", {
        "form": form,
        "result": result,
        "label": label,
        "confidence": confidence,
        "model_missing": model_missing
    })

# -------------------------------
# XGBOOST VIEW
# -------------------------------
def xgb_view(request):
    form = XGBForm(request.POST or None)
    result = label = confidence = None
    model_missing = False

    try:
        xgb_model = _load_model("xgb_model.pkl")
    except FileNotFoundError:
        xgb_model = None
        model_missing = True

    if form.is_valid() and xgb_model is not None:
        try:
            location = form.cleaned_data['location']
            skill = form.cleaned_data['skill']
            company_name = form.cleaned_data['company_name']
            platform_name = form.cleaned_data['platform_name']
            degree = form.cleaned_data['degree']

            location_freq = df['Location'].value_counts(normalize=True).get(location, 0)
            skill_score = df['skill_score'].value_counts(normalize=True).get(skill, 0)
            company_freq = df['company_name'].value_counts(normalize=True).get(company_name, 0)
            platform_freq = df['platform_name'].value_counts(normalize=True).get(platform_name, 0)
            degree_encoded = pd.factorize(df['degree'])[1].get_loc(degree) if degree in df['degree'].values else 0

            X = np.array([[location_freq, skill_score, company_freq, platform_freq, degree_encoded]])

            result = int(xgb_model.predict(X)[0])
            insurance_label = {0: "No Insurance", 1: "Insurance"}
            label = insurance_label.get(result, "Unknown")

            if hasattr(xgb_model, 'predict_proba'):
                confidence = round(float(np.max(xgb_model.predict_proba(X)[0]) * 100), 2)

        except Exception:
            logger.exception("Erreur lors de la prédiction XGBoost")
            result = label = confidence = None

    return render(request, "xgboost.html", {
        "form": form,
        "result": result,
        "label": label,
        "confidence": confidence,
        "model_missing": model_missing
    })

# -------------------------------
# REGRESSION VIEW
# -------------------------------
exp_map = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
company_size_map = {'S': 0, 'M': 1, 'L': 2}
employment_map = {'FT': 0, 'PT': 1, 'CT': 2, 'FL': 3}
DEFAULT_SALARY_MEAN = 60000.0

def _global_salary_mean():
    if not df1.empty and "salary_in_usd" in df1.columns:
        return float(df1["salary_in_usd"].mean())
    return DEFAULT_SALARY_MEAN

def get_target_encoding(col, val):
    if not df1.empty and col in df1.columns and "salary_in_usd" in df1.columns:
        te_map = df1.groupby(col)["salary_in_usd"].mean()
        return float(te_map.get(val, _global_salary_mean()))
    return _global_salary_mean()

# Load regression model from ml_models directory
try:
    salary_model = _load_model("xgb_salary_model.pkl")
    feature_order = list(salary_model.feature_names_in_) if hasattr(salary_model, "feature_names_in_") else []
except (FileNotFoundError, Exception) as e:
    logger.warning(f"Could not load xgb_salary_model.pkl: {e}")
    salary_model = None
    feature_order = []

def regression_view(request):
    form = RegressionForm(request.POST or None)
    result = error_message = None
    model_missing = salary_model is None or not feature_order

    if model_missing:
        error_message = "Le modele de regression n'est pas disponible."

    if form.is_valid() and not model_missing:
        cd = form.cleaned_data
        try:
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

            X_input = np.array([[row[f] for f in feature_order]], dtype=float)
            log_pred = salary_model.predict(X_input)[0]
            pred_salary = np.expm1(log_pred)
            result = round(float(pred_salary), 2)
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

# -------------------------------
# SKILLS GAP / RECOMMENDATION VIEW
# -------------------------------
# Normalisation des skills pour comparaison


# =========================
# UTILITY FUNCTION (IMPORTANT)
# =========================
def clean_skill(skill):
    """
    Normalize skills so user input and dataset skills match
    """
    return (
        skill.lower()
        .replace("skill_", "")
        .replace(" ", "_")
        .strip()
    )


# =========================
# MAIN VIEW
# =========================
def recommandation_view(request):
    form = RecommandationForm(request.POST or None)
    results = None
    error_message = None

    # Check models availability
    models_ready = all([
        encoder,
        classifier,
        le,
        skills_cols,
        top_skills_by_title
    ])

    if not models_ready:
        error_message = "Recommendation model is not available."

    if form.is_valid() and models_ready:
        cd = form.cleaned_data

        # -------------------------
        # USER INPUT
        # -------------------------
        target_job = cd.get("job_target", "").strip().upper()
        candidate_skills_raw = cd.get("user_skills", "")

        # Normalize user skills
        candidate_skills = [
            clean_skill(s)
            for s in candidate_skills_raw.split(",")
            if s.strip()
        ]

        candidate_set = set(candidate_skills)

        # -------------------------
        # BUILD INPUT VECTOR
        # -------------------------
        cand_vector = np.zeros(len(skills_cols))

        for skill in candidate_set:
            col_name = f"skill_{skill}"
            if col_name in skills_cols:
                idx = skills_cols.index(col_name)
                cand_vector[idx] = 1

        X_input = cand_vector.reshape(1, -1)

        # -------------------------
        # MODEL PREDICTION
        # -------------------------
        cand_embed = encoder.predict(X_input, verbose=0)
        probs = classifier.predict_proba(cand_embed)[0]

        top_idx = np.argsort(probs)[-6:][::-1]

        recommendations = []

        # -------------------------
        # BUILD RECOMMENDATIONS
        # -------------------------
        for idx in top_idx:
            job = le.inverse_transform([idx])[0]
            score = round(float(probs[idx]) * 100, 2)

            job_skills_raw = top_skills_by_title.get(job, [])
            job_skills = {clean_skill(s) for s in job_skills_raw}

            matching_skills = sorted(job_skills & candidate_set)
            missing_skills = sorted(job_skills - candidate_set)

            recommendations.append({
                "job": job,
                "score": score,
                "matching_skills": matching_skills,
                "missing_skills": missing_skills[:6],
            })

        # -------------------------
        # FORCE TARGET JOB FIRST
        # -------------------------
        recommendations.sort(
            key=lambda x: (
                x["job"].upper() != target_job,
                -x["score"]
            )
        )

        # -------------------------
        # TARGET JOB STATS
        # -------------------------
        target_rec = next(
            (r for r in recommendations if r["job"].upper() == target_job),
            None
        )

        results = {
            "target_job": target_job,
            "target_score": target_rec["score"] if target_rec else 0,
            "matching_skills": target_rec["matching_skills"] if target_rec else [],
            "missing_skills": target_rec["missing_skills"] if target_rec else [],
            "recommendations": recommendations,
        }

    elif request.method == "POST" and not form.is_valid():
        error_message = "Please correct the form errors."

    return render(request, "recommandation.html", {
        "form": form,
        "results": results,
        "error_message": error_message,
    })


# -------------------------------
# HOME / DASHBOARD
# -------------------------------
def home(request):
    return render(request, "home.html")

def dashboard(request):
    return render(request, "dashboard.html")
