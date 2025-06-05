# Analyzing Fire Response Times, Water Usage, and Incident Type Predictability in Izmir

Author: Erdi Kılıç (kilic.erdi@metu.edu.tr)  
Bioinformatics, Graduate School of Informatics  
Middle East Technical University  
Ankara, Turkey

## Description

This project analyzes the 'Izmir 2023 Fire Response Statistics' dataset to investigate key fire response metrics. The primary goals are:
* To determine if there are significant differences in average response times across various fire types.
* To assess the variability in water usage among different fire categories.
* To explore the feasibility of predicting fire incident types using machine learning.

The insights from this analysis aim to inform predictive modeling and resource allocation for more effective fire management.

## Key Features

* **Data Preprocessing:** Includes data loading, cleaning (handling missing values, dropping irrelevant columns), feature engineering (creating aggregate casualty counts, mapping fire incident types to broader categories), and feature transformation (scaling numerical features, one-hot encoding categorical features ). A custom transformation was applied to the team departure time feature.
* **Statistical Analysis:** Employs statistical tests to analyze differences in fire response times and water usage across fire types.
* **Predictive Modeling:**
    * Develops a CatBoost classifier to predict fire incident types.
    * Includes hyperparameter tuning using `RandomizedSearchCV` with 3-fold cross-validation, guided by the $F1_{macro}$ score.
    * Evaluates model performance against a `DummyClassifier` baseline.
    * Visualizes feature importances from the CatBoost model.
* **Visualization:** Generates various plots including feature importance bar charts (Fig. 1 in PDF), confusion matrices (Fig. 3 in PDF), and ROC curves (Fig. 2 in PDF).

---
## Statistical Tests and Findings

### Fire Response Times (VARIS_SURESI)
* **Objective:** To determine if average emergency response times vary across different fire incident categories (YANGIN_TURU).
* **Normality Check:** The Shapiro-Wilk test indicated that response time data did not conform to a normal distribution for any fire type group (all $p<.05$).
* **Homogeneity of Variances:** Levene's test was statistically significant ($W(5,N_{total}-6)=3.987, p=.001292$), leading to the rejection of the null hypothesis of equal variances. This signifies that variability in response times differs significantly among fire categories.
* **Mean Comparison:** Due to non-normality and heteroscedasticity, Welch's ANOVA was used. The test yielded a statistically significant result ($F(5,550.790612)= 60.676, p<.0001$), indicating a significant overall difference in mean response times. The partial eta-squared $(\eta_{p}^{2}=.0041)$ showed fire type accounts for a small but significant proportion of variance in response times.
* **Post-Hoc Analysis:** The Games-Howell test revealed statistically significant pairwise differences. For example:
    * "Bitki Örtüsü - Tarımsal Alan Yangınları" (mean=6.97) had significantly longer response times than "Altyapı Ekipman Yangınları" (mean=5.18, $p<.0001$) and "Araç Yangınları" (mean=5.49, $p<.0001$).
    * "Yapı Mesken Yangınları" (mean=4.98) had significantly shorter response times compared to "Araç Yangınları" (mean=5.49, $p=.003$) and "Bitki Örtüsü Tarımsal Alan Yangınları" (mean=6.97, $p<.0001$).
* **Conclusion:** Average emergency response times vary significantly across different types of fire incidents.

### Water Usage (KULLANILAN_SU_MIKTARI)
* **Objective:** To determine if average water quantity used varies across different fire incident classifications.
* **Normality Check:** The Shapiro-Wilk test indicated that water usage data also departed from a normal distribution for all categories ($p<.05$).
* **Homogeneity of Variances:** Levene's test was statistically significant ($W(5,N_{total}-6)= 11.162, p<.0001$), signifying unequal variability in water consumption among fire types.
* **Mean Comparison:** Welch's ANOVA was employed, yielding a highly statistically significant outcome ($F(5,543.095)=24.071, p<.0001$). The partial eta-squared value $(\eta_{p}^{2}=.004128)$ indicated the proportion of variance in water usage attributable to fire type.
* **Post-Hoc Analysis:** The Games-Howell test revealed several notable distinctions. For example:
    * "Yapı Mesken Yangınları" (mean = 5.89 units) required significantly more water than "Altyapı - Ekipman Yangınları" (mean=1.02 units, $p<.0001$) and "Araç Yangınları" (mean=1.33 units, $p<.0001$).
    * "Bitki Örtüsü - Tarımsal Alan Yangınları" (mean=4.24 units) also showed significantly greater mean water usage compared to "Altyapı Ekipman Yangınları" ($p<.0001$) and "Araç Yangınları" ($p<.0001$).
* **Conclusion:** The specific category of a fire incident is a significant factor influencing the volume of water consumed.

---
## Modeling Results

### Classification Metrics Summary
A CatBoost classifier was developed and compared against a stratified DummyClassifier baseline.

| Metric                  | Baseline   | CatBoost   |
| :---------------------- | :--------- | :--------- |
| Accuracy                | 0.26       | 0.81       |
| Macro Avg. Precision    | 0.18       | 0.75       |
| Macro Avg. Recall       | 0.19       | 0.74       |
| Macro Avg. F1-score     | 0.1844     | 0.7328     |
| Weighted Avg. Precision | 0.26       | 0.81       |
| Weighted Avg. Recall    | 0.26       | 0.81       |
| Weighted Avg. F1-score  | 0.26       | 0.81       |
| MCC                     | -0.0226    | 0.7350     |

*Table data sourced from *

**Baseline Model (DummyClassifier):**
* Exhibited performance characteristic of random guessing.
* Overall accuracy: 0.26.
* Macro average F1-score: 0.1844.
* MCC: -0.0226, indicating no meaningful correlation.
* AUC values for most classes hovered around 0.50.

**CatBoost Classifier:**
* Demonstrated substantially superior performance across all metrics and classes.
* Overall accuracy: 0.81.
* Macro average F1-score: 0.7328.
* MCC: 0.7350, signifying a strong positive correlation.
* AUC values were consistently high across all fire categories, e.g., "Altyapı - Ekipman Yangınları" (AUC=0.96), "Araç Yangınları" (AUC=0.98), and "Yapı Mesken Yangınları" (AUC=1.00).
* The CatBoost model improved accuracy by 211% relative to the baseline.

**Confusion Matrix (CatBoost):**
* Generally high predictive accuracy across the six fire categories.
* Exceptional accuracy for "Yapı - Mesken Yangınları" (309/311 correctly identified).
* Robust classification for "Araç Yangınları" (95/110) and "Bitki Örtüsü Tarımsal Alan Yangınları" (400/499).
* Some misclassifications were noted, particularly between "Atık Hurda Depolanmış Malzeme Yangınları" and "Bitki Örtüsü - Tarımsal Alan Yangınları".

### Feature Importance (CatBoost)
The most influential features for the CatBoost model were:
1.  `YAPI_SEKLI` (Building Structural Type) 
2.  `YANGIN_SEBEBI` (Fire Cause) 
3.  `ILCE` (District) 
Other notable features included `KULLANILAN_SU_MIKTARI` (Amount of Water Used), `EKIPLERIN_CIKIS_SAATI` (Team Departure Time), and `ADRES_BOLGESI` (Address Region). These findings are consistent with literature where structural characteristics, ignition source, and geographical location are significant determinants.

---
## Dataset

The project utilizes the `2023_yili_yangin_mudahale_istatistigi.csv` dataset, which contains fire response statistics for Izmir in 2023. A cleaned version, `2023_yili_yangin_mudahale_istatistigi_cleaned.csv`, is generated during preprocessing.

## Libraries Used

* numpy
* pandas
* matplotlib
* seaborn
* scikit-learn (for model selection, preprocessing, dummy classifier, metrics)
* scipy (for statistical tests)
* pingouin (for statistical tests like ANOVA, normality, homoscedasticity)
* shap (imported but not explicitly used in the provided notebook snippet for core analysis)
* catboost (for the CatBoost classifier)
* warnings

## Code Structure

The primary analysis is contained within the Jupyter Notebook: `erdikilic_project_deliverable.ipynb`.
The notebook is structured as follows:
1.  **Imports and Setup:** Loading necessary libraries and initial configurations.
2.  **Data Preprocessing:**
    * Function to convert time strings to minutes.
    * Loading the raw dataset.
    * Data cleaning, feature elimination, and feature generation.
    * Mapping of `YANGIN_TURU` to broader categories.
3.  **Feature Importance (Initial Check):** Pre-training a CatBoost model.
4.  **Data Splitting and Further Preprocessing:** Using `ColumnTransformer`.
5.  **Baseline Model (DummyClassifier):** Training and evaluation.
6.  **CatBoost Model Development:** Hyperparameter tuning and final model training.
7.  **Model Evaluation:** Metrics, confusion matrices, and ROC curves.
8.  **Statistical Tests:** Normality, homogeneity of variance checks, ANOVA, and post-hoc tests.

## How to Run

1.  Ensure you have Python and the listed libraries installed.
2.  Place the dataset `2023_yili_yangin_mudahale_istatistigi.csv` in a `../data/` directory relative to the notebook, or update the path in the notebook.
3.  Open and run the `erdikilic_project_deliverable.ipynb` notebook in a Jupyter environment.
    * The notebook will perform data preprocessing, train models, evaluate them, and conduct statistical tests.
    * Figures generated will be displayed and saved to the `../reports/figures/` directory.
    * A cleaned version of the dataset will be saved to `../data/2023_yili_yangin_mudahale_istatistigi_cleaned.csv`.

## References (from the paper)  
[1] C.-Y. Ku and C.-Y. Liu, "Predictive modeling of fire incidence using deep neural networks," Fire, vol. 7. no. 4, p. 136. Apr. 2024. doi: 10.3390/fire7040136.  
[2] Y. Yuan and A. G. Wylie, "Comparing machine learning and time series approaches in predictive modeling of urban fire incidents: A case study of austin, Texas," ISPRS International Journal of Geo-Information, vol. 13, no. 5. p. 149, Apr. 2024. doi: 10.3390/ijgi13050149.  
[3] A. Schmidt, E. Gemmil, and R. Hoskins, Machine learning based risk analysis and predictive modeling of structure fire related casualties, 2024. doi:10.2139/ssrn.4757205.  
[4] G. V. Kuznetsov, A. O. Zhdanova, R. S. Volkov, and P. A. Strizhak, "Optimizing firefighting agent consumption and fire suppression time in buildings by forming a fire feedback loop," Process Safety and Environmental Protection, vol. 165, pp. 754-775. Sep. 2022. doi:10.1016/j.psep.2022.07.061.

---

*Disclaimer: This README is based on the provided Jupyter Notebook and research paper. The "shap" library was imported but its specific use for detailed SHAP value plots was not in the executed cells shown; it might be intended for further model interpretability analysis.*