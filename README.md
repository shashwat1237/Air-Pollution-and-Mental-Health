📘 README.md — Mental State, Environment & Behaviour Analysis
🧠 Understanding Wellbeing Through Behaviour, Environment & Physiology

This project explores how daily wellbeing, sleep, energy, and stress relate to:

Environmental exposures (air pollution, weather, noise)

Cognitive performance (Stroop task metrics)

Daily behaviour (steps, mobile use, activity)

Demographics & time variables

The notebook performs cleaning, EDA, clustering, supervised ML, and SHAP explainability to understand drivers of daily mental state.

📁 Project Structure
├── data/              # Raw and/or cleaned dataset (not included in repo)
├── notebook.ipynb     # Main analysis notebook
├── README.md          # Project documentation (this file)

🧹 1. Data Cleaning Pipeline

The dataset originally contained ~56 numeric features with ~1,594 non-null rows per feature.

Cleaning steps included:

Dropping irrelevant columns (ID_Zenodo)

Removing rows with missing values (dropna())

Outlier removal using IQR on key psychological targets:

sueno, estres, energia, bienestar

Standardizing data using StandardScaler

Binning target variables into balanced classes using medians

Final cleaned dataset size: ~1300–1400 rows, 55 numeric features.

🔎 2. Exploratory Data Analysis (EDA)
Key visual and statistical techniques:

Distribution inspection (describe(), histograms, boxplots)

Correlation analysis:

absolute & signed matrices

melted top 200 correlations

pollutant–weather relationships

Behavioural and cognitive performance analysis

Environmental feature mapping (PM2.5, NO₂, BC, humidity, wind, precipitation)

Mental state distribution analysis (sleep, energy, stress, wellbeing)

🧩 3. Understanding the Data
Mental state variables are highly correlated

energia ↔ bienestar: +0.67

sueno ↔ bienestar: +0.39

estres ↔ bienestar: −0.34

Environmental exposures show weak but logical trends

PM2.5 decreases with precipitation & wind
(correlations around −0.30)

NO₂, BC, PM₂.₅ strongly inter-correlated (0.82+)

Cognitive metrics behave realistically

Older age → slower response times → lower z-performance

Stroop congruent & incongruent times correlate strongly (0.76)

Heavy redundancy in features

12h vs 24h weather: r ≈ 0.95–0.99

seconds vs hours versions: r ≈ 1.00

scaled (×30) vs original pollutant values: r ≈ 1.00

This justifies later use of feature importance and SHAP.

🧭 4. Clustering Results

K-Means tested for k = 2 to 10

Silhouette scores ranged 0.09–0.12

Indicates no strong cluster structure

Dataset is a continuous spectrum of behaviours and exposures

🤖 5. Machine Learning — Predicting Mental States

The function f(column_to_predict) builds end-to-end models for:

Sleep Quality (sueno)

Stress (estres)

Energy (energia)

Wellbeing (bienestar)

ML pipeline:

Balanced binning via median splits

Train-test split (80/20, stratified)

Oversampling to address class imbalance

XGBoost Classifier

Evaluation using:

AUC-PR

Average Precision (AP)

Accuracy

Precision/Recall/F1

Confusion Matrix

🔥 Typical performance:

PR-AUC: 0.68–0.70

Accuracy: 0.60–0.66

Good recall for “low state” cases

Harder to predict “high state” days

📊 6. Explainability (XGBoost + SHAP)

SHAP was used to understand:

Which variables influence each target?

Direction of impact (positive/negative)?

Contribution to classification decisions?

This allows transparent interpretation of mental-state prediction models.

💡 7. Key Insights From Modeling

Internal variables dominate predictability:
Energy, sleep, and stress strongly determine wellbeing.

Environment matters but weakly linearly:
Pollution/weather signals exist but are subtle.

Behavioural metrics contribute moderately:
Steps, activity times, mobile use show small but real effects.

Cognitive performance correlates with age and stress:
Slower reaction times predict lower wellbeing/energy.

Multicollinearity is high, but XGBoost handles it well:
PCA was intentionally not used to preserve interpretability.

🚀 8. Why PCA Was Not Used (and why that’s correct)

PCA was intentionally excluded because:

XGBoost handles multicollinearity automatically

PCA destroys interpretability

PCA removes non-linear structure important in mental-state modeling

PCA is unnecessary with <100 features

Avoiding PCA was a correct and optimal choice.

🗂 9. Technologies Used

Python 3

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

XGBoost

SHAP

📌 10. How to Run
pip install -r requirements.txt
jupyter notebook


Open the notebook and run all cells sequentially.

🧾 11. Future Improvements

Feature selection to remove redundant variables

Hyperparameter tuning for XGBoost

Testing LightGBM & CatBoost

Adding markdown sections directly in the notebook

AutoML benchmarking
