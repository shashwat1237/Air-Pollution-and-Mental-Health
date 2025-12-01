📘 Mental State, Environment & Behaviour Analysis
🧠 Understanding Wellbeing Through Behaviour, Environment & Physiology

This project explores how daily wellbeing, sleep, energy, and stress relate to:

Environmental exposures (air pollution, weather, noise)

Cognitive performance (Stroop task metrics)

Daily behaviour (steps, mobile use, activity)

Demographics & time variables

The notebook performs data cleaning, EDA, clustering, machine learning, and SHAP explainability to understand drivers of mental state.

📁 Project Structure
├── data/              # Raw/cleaned dataset (optional)
├── notebook.ipynb     # Main analysis notebook
├── README.md          # Project documentation

🧹 1. Data Cleaning Pipeline

The dataset originally contained ~56 numeric features with ~1,594 non-null values each.

Cleaning steps:

Dropped irrelevant columns

Removed missing values using dropna()

Removed outliers using IQR on:

sueno, estres, energia, bienestar

Standardized features using StandardScaler

Binned targets into balanced classes using medians

Final dataset size: ~1300–1400 rows, 55 numeric features.

🔎 2. Exploratory Data Analysis (EDA)
Techniques used:

Value distributions & boxplots

Summary statistics (describe())

Correlation analysis (absolute and signed)

Top 200 strongest correlations

Environmental–behavioural analysis

Cognitive performance patterns

Mental state distributions

🧩 3. Key Findings From the Data
Mental State Relationships

bienestar ↔ energia: +0.67

bienestar ↔ sueno: +0.39

bienestar ↔ estres: −0.34

Energy is the strongest driver of wellbeing.

Environmental Trends

PM2.5 reduces with rain & wind (−0.30)

All pollutants are highly intercorrelated (0.82+)

Behaviour & Cognition

Older age → slower cognitive respones → lower z-performance

Stroop congruent & incongruent times correlate (0.76)

Redundant Features

Seconds ↔ hours = 1.00 correlation

12h vs 24h weather = 0.95–0.99 correlation

Scaled vs unscaled pollutants = 1.00 correlation

🧭 4. Clustering Results

K-Means tested from k = 2 to 10.
Silhouette scores: 0.09–0.12

➡️ The dataset does not contain strong clusters.
Mental states vary continuously, not in distinct groups.

🤖 5. Machine Learning Pipeline

The function f(column_to_predict) builds a model for:

Sleep Quality (sueno)

Stress (estres)

Energy (energia)

Wellbeing (bienestar)

Steps:

Median binning → Balanced classes

Train-test split (80/20, stratified)

Oversampling for balance

XGBoost classifier

Evaluation using:

AUC–PR

Average Precision

Accuracy

Precision, Recall, F1

Confusion matrix

Typical Performance

AUC–PR: ~0.68–0.70

Accuracy: 0.60–0.66

High recall for “low state” classes

Harder to detect “high state” classes

📊 6. Explainability (SHAP)

SHAP was used to identify:

Most influential variables

Their direction (positive/negative impact)

Their contribution to model predictions

TreeExplainer + summary plots provided model interpretability.

💡 7. Key Insights From Modeling

Internal variables (sleep, stress, energy) are the strongest predictors

Environment contributes but weakly

Behavioural data adds moderate signal

Cognitive performance correlates with mental state

XGBoost handles multicollinearity well

🚀 8. Why PCA Was Not Used

PCA was intentionally not applied because:

XGBoost handles multicollinearity automatically

PCA harms interpretability

PCA removes non-linear structure

Dataset size is small (55 features) → PCA unnecessary

Avoiding PCA was the correct choice.

🗂 9. Technologies Used

Python 3

Pandas, NumPy

Matplotlib

Scikit-learn

XGBoost

SHAP

📌 10. How to Run
pip install -r requirements.txt
jupyter notebook


Open notebook.ipynb and run all cells sequentially.

🧾 11. Future Improvements

Feature selection (remove redundant variables)

Hyperparameter tuning for XGBoost

Trying LightGBM/CatBoost

Adding markdown explanation within the notebook

Automated pipeline
