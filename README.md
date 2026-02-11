# API FastAPI - Pipeline ML Complet (5 TPs)

API FastAPI, incluant nettoyage de données, analyse exploratoire, analyse multivariée, ML baseline et ML avancé.

## Fonctionnalités

- TP1 - Clean API : Nettoyage de données (NA, duplicates, outliers)
- TP2 - EDA API : Analyse exploratoire avec visualisations Plotly
- TP3 - MV API : PCA et Clustering (K-means)
- TP4 - ML API : Entraînement et prédiction (LogReg, RF)
- TP5 - ML Avancé : Tuning (GridSearch/RandomSearch) et Explicabilité

## Démarrage rapide avec Docker

### Prérequis
- Docker Desktop installé

### Lancer l'API
bash
docker-compose up -d

### Accéder à la documentation

http://localhost:8000/docs

### Arrêter l'API
bash
docker-compose down

## Installation sans Docker

### Prérequis
- Python 3.12+

### Installation
bash
pip install -r requirements.txt

### Lancer l'API
bash
uvicorn app.main:app --reload

## Documentation

Consultez les rapports détaillés :
- [TP1_Rapport.adoc](TP1_Rapport.adoc)
- [TP2_Rapports.adoc](TP2_Rapports.adoc)
- [TP3_Rapport.adoc](TP3_Rapport.adoc)
- [TP4_Rapport.adoc](TP4_Rapport.adoc)
- [TP5_Rapports.adoc](TP5_Rapports.adoc)

## Technologies

- FastAPI
- Pandas, NumPy
- Scikit-learn
- Plotly
- Docker

## Licence

Projet académique