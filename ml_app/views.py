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
    return render(request, 'dashboard.html')


def dashboard(request):
    return render(request, 'dashboard.html')


'''def regression_view(request):
    form = RegressionForm(request.POST or None)
    result = None
    model_missing = False
    error_message = None

    try:
        poly_model = _load_model('poly_pipeline_degree.pkl')
    except FileNotFoundError:
        poly_model = None
        model_missing = True

    if form.is_valid() and poly_model is not None:
        try:
            # Récupérer les valeurs du formulaire
            experience_level = form.cleaned_data['experience_level']
            company_size = form.cleaned_data['company_size']
            employment_type = form.cleaned_data['employment_type']
            work_year = form.cleaned_data['work_year']
            employee_residence = form.cleaned_data['employee_residence']
            company_location = form.cleaned_data['company_location']
            remote_ratio = form.cleaned_data['remote_ratio']
            job_title = form.cleaned_data['job_title']
            salary_currency = form.cleaned_data['salary_currency']

            # Encodage
            exp_map = {'EN':0, 'MI':1, 'SE':2, 'EX':3}
            company_size_map = {'S':0, 'M':1, 'L':2}
            employment_map = {'FT':0, 'PT':1, 'CT':2, 'FL':3}

            experience_level_enc = exp_map.get(experience_level, 0)
            company_size_enc = company_size_map.get(company_size, 0)
            employment_type_enc = employment_map.get(employment_type, 0)

            # Fréquences — gérer colonnes manquantes dans `df1`
            def _freq(col, val):
                if col in df1.columns:
                    return df1[col].value_counts(normalize=True).get(val, 0)
                logger.debug("Column '%s' not found in df1, returning 0 for frequency", col)
                return 0

            work_year_freq = _freq('work_year', work_year)
            employee_residence_freq = _freq('employee_residence', employee_residence)
            company_location_freq = _freq('company_location', company_location)
            remote_ratio_freq = _freq('remote_ratio', remote_ratio)
            job_title_freq = _freq('job_title', job_title)
            salary_currency_freq = _freq('salary_currency', salary_currency)

            # Préparer X
            X = np.array([[experience_level_enc, company_size_enc, employment_type_enc,
                           work_year_freq, employee_residence_freq, company_location_freq,
                           remote_ratio_freq, job_title_freq, salary_currency_freq]])
            
            # Prédiction
            result = poly_model.predict(X)[0]

        except Exception as e:
            logger.exception("Erreur lors de la prédiction Regression")
            result = None
            error_message = str(e)

    return render(request, 'regression.html', {
        'form': form,
        'result': result,
        'model_missing': model_missing,
        'error_message': error_message,
    })'''
'''def regression_view(request):
    form = RegressionForm(request.POST or None)
    result = None
    model_missing = False
    error_message = None

    # Charger le modèle Polynomial Regression
    try:
        poly_model = _load_model('poly_pipeline_degree.pkl')
    except FileNotFoundError:
        poly_model = None
        model_missing = True

    if form.is_valid() and poly_model is not None:
        try:
            # --- Récupérer les valeurs du formulaire ---
            experience_level = form.cleaned_data['experience_level']
            company_size = form.cleaned_data['company_size']
            employment_type = form.cleaned_data['employment_type']
            work_year = form.cleaned_data['work_year']
            employee_residence = form.cleaned_data['employee_residence']
            company_location = form.cleaned_data['company_location']
            remote_ratio = form.cleaned_data['remote_ratio']
            job_title = form.cleaned_data['job_title']
            salary_currency = form.cleaned_data['salary_currency']

            # --- Encodage ---
            exp_map = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
            company_size_map = {'S': 0, 'M': 1, 'L': 2}
            employment_map = {'FT': 0, 'PT': 1, 'CT': 2, 'FL': 3}

            experience_level_enc = exp_map.get(experience_level, 0)
            company_size_enc = company_size_map.get(company_size, 0)
            employment_type_enc = employment_map.get(employment_type, 0)

            # --- Préparer les fréquences en évitant les zéros ---
            def get_freq(col, val):
                freq_col = col + "_freq" if col + "_freq" in df1.columns else None
                if freq_col:
                    return df1[freq_col].get(val, 0.0001)  # minimum pour éviter NaN ou 0
                return 0.0001

            work_year_freq = get_freq('work_year', work_year)
            employee_residence_freq = get_freq('employee_residence', employee_residence)
            company_location_freq = get_freq('company_location', company_location)
            remote_ratio_freq = get_freq('remote_ratio', remote_ratio)
            job_title_freq = get_freq('job_title', job_title)
            salary_currency_freq = get_freq('salary_currency', salary_currency)

            # --- Préparer le tableau X pour prédiction ---
            X = np.array([[
                experience_level_enc,
                company_size_enc,
                employment_type_enc,
                work_year_freq,
                employee_residence_freq,
                company_location_freq,
                remote_ratio_freq,
                job_title_freq,
                salary_currency_freq
            ]], dtype=float)

            # --- Prédiction ---
            result = poly_model.predict(X)[0]

        except Exception as e:
            logger.exception("Erreur lors de la prédiction Regression")
            result = None
            error_message = str(e)

    return render(request, 'regression.html', {
        'form': form,
        'result': result,
        'model_missing': model_missing,
        'error_message': error_message,
    })'''
def regression_view_debug(request):
    form = RegressionForm(request.POST or None)
    result = None
    model_missing = False
    error_message = None
    X_debug = None  # pour affichage

    # Charger le modèle Polynomial Regression
    try:
        poly_model = _load_model('poly_pipeline_degree.pkl')
    except FileNotFoundError:
        poly_model = None
        model_missing = True

    if form.is_valid() and poly_model is not None:
        try:
            # Récupérer les valeurs du formulaire
            experience_level = form.cleaned_data['experience_level']
            company_size = form.cleaned_data['company_size']
            employment_type = form.cleaned_data['employment_type']
            work_year = form.cleaned_data['work_year']
            remote_ratio = form.cleaned_data['remote_ratio']
            employee_residence = form.cleaned_data['employee_residence']
            company_location = form.cleaned_data['company_location']
            job_title = form.cleaned_data['job_title']
            salary_currency = form.cleaned_data['salary_currency']

            # Encodage
            exp_map = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
            company_size_map = {'S': 0, 'M': 1, 'L': 2}
            employment_map = {'FT': 0, 'PT': 1, 'CT': 2, 'FL': 3}

            experience_level_enc = exp_map.get(experience_level, 0)
            company_size_enc = company_size_map.get(company_size, 0)
            employment_type_enc = employment_map.get(employment_type, 0)

            # Fréquences exactes
            def get_freq(col, val):
                freq_col = col + "_freq"
                if freq_col in df1.columns:
                    row = df1[df1[col] == val]
                    if not row.empty:
                        return float(row[freq_col].iloc[0])
                return 0.0001

            work_year_freq = get_freq('work_year', work_year)
            employee_residence_freq = get_freq('employee_residence', employee_residence)
            company_location_freq = get_freq('company_location', company_location)
            remote_ratio_freq = get_freq('remote_ratio', remote_ratio)
            job_title_freq = get_freq('job_title', job_title)
            salary_currency_freq = get_freq('salary_currency', salary_currency)

            # Tableau X (9 features)
            X = np.array([[  
                experience_level_enc,
                company_size_enc,
                employment_type_enc,
                work_year_freq,
                employee_residence_freq,
                company_location_freq,
                remote_ratio_freq,
                job_title_freq,
                salary_currency_freq
            ]], dtype=float)

            X_debug = X.copy()
            print("Features passées au modèle:", X_debug)

            # Prédiction
            result = poly_model.predict(X)[0]

        except Exception as e:
            logger.exception("Erreur lors de la prédiction Regression")
            result = None
            error_message = str(e)

    return render(request, 'regression.html', {
        'form': form,
        'result': result,
        'model_missing': model_missing,
        'error_message': error_message,
        'X_debug': X_debug,  # pour debug dans template si besoin
    })


def regression_view(request):
    """Compatibility wrapper used by urls.py. Calls the debug implementation."""
    return regression_view_debug(request)

def kmeans_view(request):
    return render(request, 'kmeans.html')
