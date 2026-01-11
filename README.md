# 🏥 Diabetes Prediction Challenge (Kaggle)

My solution for the [Playground Series S5E12](https://www.kaggle.com/competitions/playground-series-s5e12) competition.
The goal is to predict whether a patient has diabetes based on their medical records and demographic data.

**🏆 Best Public Score (ROC-AUC):** `0.6969

## 🛠️ Tech Stack
* **Language:** Python 3.12
* **Libraries:** Pandas, Scikit-learn, LightGBM, CatBoost
* **Tools:** VS Code, Jupyter Notebook

## Approach & Strategy

### 1. Data Preprocessing
* Handled categorical variables (`gender`, `smoking_status`) using LightGBM's native support and CatBoost encoding.
* Analyzed data distribution and checked for class imbalance.

### 2. Feature Engineering
Created new features to capture non-linear relationships:
* **Risk Factor:** `Age * BMI` (Captures the cumulative risk of obesity with age).
* **Medical Indices:** Calculated `Cholesterol_Ratio` (LDL/HDL) and Pulse Pressure.

### 3. Modeling
I used an ensemble of Gradient Boosting models:
* **LightGBM:** Tuned to prevent overfitting (`learning_rate=0.05`, `max_depth=6`).
* **CatBoost:** Utilized for better handling of categorical features (`iterations=1000`).

### 4. Ensemble (Blending)
The final submission is a weighted blend of predictions:
* `60%` LightGBM
* `40%` CatBoost
* This approach improved the ROC-AUC score by smoothing out individual model errors.

## 📂 Project Structure
* `start.ipynb` - Main notebook with EDA, training, and blending logic.
* `train.csv` / `test.csv` - Dataset files.
* `submission_blend.csv` - Final predictions for Kaggle.

---
*Created by Timur Kim*
