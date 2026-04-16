# DS-340W-Real-Estate-Predictor
Your House Price Prediction Model documentation is ready. I have converted the text into a clean, professional GitHub-style **README.md** file.


```markdown
# 🏠 House Price Prediction Model

A machine learning project that predicts residential home sale prices using the **Kaggle House Prices: Advanced Regression Techniques** dataset. This project combines preprocessing, feature engineering, ensemble learning, and explainable AI to achieve high prediction accuracy.

## 📌 Project Overview
The goal of this project is to predict house sale prices based on housing characteristics, neighborhood information, and engineered features. The script automates the end-to-end pipeline:
- Loading Kaggle train/test datasets.
- Integrating external neighborhood and school-related features.
- Comparing multiple machine learning models.
- Selecting the best-performing model based on RMSE.
- Generating a submission-ready CSV file.

## 🤖 Models Used
The project evaluates and compares the following algorithms:
* Linear Regression
* Random Forest Regressor
* Extra Trees Regressor
* Gradient Boosting Regressor
* XGBoost Regressor
* PCA + Ridge Regression
* **Hybrid Tree Stacking Ensemble** (Best Model Selection)

## 🧠 Feature Engineering
To improve accuracy, the `add_features()` function creates several derived variables, including:
* `HouseAge`, `RemodAge`, `GarageAge`
* `TotalSF` (Total Square Footage)
* `TotalBath` (Combined full and half baths)
* `OutdoorSF` (Porch/Deck areas)
* `QualityLivArea` (Interaction between Quality and Living Area)
* `School_Score_External`, `Median_Income`, `Dist_CityHall_Miles`

## 📂 Required Files
Place the following files in your project directory:
* `train.csv`
* `test.csv`
* `ISPP_School_Summary.xlsx`
* `code2.py`

## ⚠️ Configuration: Change File Paths
Before running the script, open `code2.py` and update the `SCHOOL_PATH` variable to match your local environment.

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "test.csv")

# UPDATE THIS LINE:
SCHOOL_PATH = "/Users/YourName/Downloads/ISPP_School_Summary.xlsx"
```

## 💻 Installation
Install the necessary Python libraries using pip:

```bash
pip install pandas numpy matplotlib scikit-learn xgboost shap openpyxl
```

## ▶️ Step-by-Step: How to Run
1.  **Download Dataset:** Get `train.csv` and `test.csv` from the Kaggle House Prices competition.
2.  **Prepare External Data:** Ensure `ISPP_School_Summary.xlsx` is available.
3.  **Setup Folder:** Place the Python script and all data files in the same folder.
4.  **Edit Path:** Update `SCHOOL_PATH` in `code.py`.
5.  **Run Script:**
    ```bash
    python code.py
    ```
6.  **Wait for Training:** The script will process data, remove outliers, train models, and generate plots.

## 📊 Outputs Generated
All results are automatically saved to the `/housing_outputs/` folder:
* `model_comparison.csv` — Performance metrics for all models.
* `feature_importance.csv` — Ranking of most influential variables.
* `residual_plot.png` — Analysis of prediction errors.
* `actual_vs_pred_best_model.png` — Visual fit of the model.
* `submission_real_school_external_model.csv` — Final Kaggle submission file.
* `shap_summary.png` — Feature impact explanations (if SHAP is installed).

## 🚀 Highlights
* **Advanced Feature Engineering:** Goes beyond raw data to create meaningful indicators.
* **Ensemble Learning:** Uses stacking to combine the strengths of different models.
* **Explainable AI:** Uses SHAP values to explain *why* a specific price was predicted.
* **Automated Workflow:** From raw CSV to final submission in one click.

---
**Author:** Khang Le  and Varun
*Applied Data Science Student | Pennsylvania State University*
```
