# Analyzing Fire Response Times, Water Usage, and Incident Type Predictability in Izmir

Author: Erdi Kılıç (kilic.erdi@metu.edu.tr)  
Bioinformatics, Graduate School of Informatics  
Middle East Technical University  
Ankara, Turkey

---

1. Project Overview  
This project analyzes the 'Izmir 2023 Fire Response Statistics' dataset to investigate critical aspects of fire response. The primary objectives are:

- To determine if there are significant differences in average response times across various fire types.

- To assess the extent of variability in water usage among different fire categories.

- To evaluate the feasibility of predicting fire incident types using machine learning.

The findings aim to provide insights that can inform predictive modeling and resource allocation for more effective fire management. This research draws inspiration from existing literature on predictive modeling of fire incidents, comparison of machine learning and time series approaches, and optimization of fire suppression systems.

---

2. Dataset  
The study utilizes the 'Izmir 2023 Fire Response Statistics' dataset. This dataset contains detailed information about fire incidents in Izmir during 2023, including fire type, cause, response time, water usage, and casualty statistics.

The raw dataset (2023_yili_yangin_mudahale_istatistigi.csv) is preprocessed to prepare it for analysis. The cleaned dataset is saved as 2023_yili_yangin_mudahale_istatistigi_cleaned.csv.

---

3. Methodology

3.1. Data Preprocessing  
The data preprocessing pipeline includes:

- Data Quality Assessment: Examining data types, descriptive statistics, and visual inspection.

- Handling Missing Values: Imputing missing building structure types with "YAPI_DEGIL" (NOT_A_BUILDING) and removing rows with other missing values. Casualty counts' missing values were filled with zero before aggregation.

- Feature Engineering & Elimination:  
  - Elimination of irrelevant columns (e.g., identifiers, detailed date breakdowns, granular casualty stats).  
  - Generation of aggregate features (total human fatalities, total injuries, total animal fatalities).  
  - Mapping granular fire incident types to broader classifications.  
  - Conversion of time-related features (e.g., EKIPLERIN_CIKIS_SAATI) into minutes and applying a custom transformation.

- Feature Importance Analysis: Using a CatBoost classifier to identify influential features.

- Data Transformation:  
  - Standardization of numerical features (arrival time, suppressant quantities, total casualty counts).  
  - One-hot encoding of categorical features (fire cause, building type, district, address region, fire outcome).

3.2. Statistical Analysis  
Two primary statistical investigations were conducted:

- Fire Response Times (VARIS_SURESI):  
  Null Hypothesis (H₀): The mean response times are the same across all fire types.  
  Alternative Hypothesis (Hₐ): At least one fire type has a different mean response time.  
  Non-normality (Shapiro-Wilk test) and heteroscedasticity (Levene's test) were observed.  
  Welch's ANOVA was used to compare means, followed by the Games-Howell post-hoc test for pairwise comparisons.

- Water Usage (KULLANILAN_SU_MIKTARI):  
  Null Hypothesis (H₀): The mean water usage is the same across all fire types.  
  Alternative Hypothesis (Hₐ): At least one fire type has a different mean water usage.  
  Similar to response times, non-normality and heteroscedasticity were found.  
  Welch's ANOVA and Games-Howell post-hoc test were employed.

3.3. Predictive Modeling  
Target Variable: Fire Incident Type (YANGIN_TURU) after mapping to broader categories.

Algorithm: CatBoost Classifier, chosen for its proficient handling of categorical data and robustness against overfitting.

Baseline Model: DummyClassifier with a stratified strategy.

Hyperparameter Tuning: RandomizedSearchCV with 3-fold cross-validation, using the F1_macro score as the guiding metric. The test set was split into a validation set (for tuning) and a final held-out test set.

Evaluation Metrics: Accuracy, Macro Average Precision/Recall/F1-score, Weighted Average Precision/Recall/F1-score, Matthews Correlation Coefficient (MCC), and Area Under the ROC Curve (AUC) per class.

Interpretability: CatBoost's built-in feature importance scores were utilized.

---

4. Key Findings

4.1. Statistical Analysis  
Fire Response Times: Statistically significant variations exist in average emergency response times across different fire types (Welch's ANOVA: F(5,550.79)=60.676,p<.0001,η_p^2=.0041).

For example, "Bitki Örtüsü - Tarımsal Alan Yangınları" (Vegetation - Agricultural Area Fires) had significantly longer mean response times compared to "Altyapı Ekipman Yangınları" (Infrastructure Equipment Fires) and "Araç Yangınları" (Vehicle Fires).

Water Usage: Statistically significant differences exist in the average quantity of water used across various fire incident classifications (Welch's ANOVA: F(5,543.095)=24.071,p<.0001,η_p^2=.004128).

For example, "Yapı Mesken Yangınları" (Building - Residential Fires) required significantly more water on average than infrastructure, vehicle, and waste material fires.

4.2. Predictive Modeling  
The CatBoost classifier significantly outperformed the baseline model.

CatBoost Accuracy: 0.81

CatBoost Macro Avg. F1-score: 0.7328

CatBoost MCC: 0.7350

AUC scores for the CatBoost model were consistently high across all fire categories (e.g., 0.96 for Infrastructure Equipment Fires, 0.98 for Vehicle Fires, and nearly 1.00 for Building - Residential Fires).

Some misclassifications were noted, particularly between "Atık Hurda Depolanmış Malzeme Yangınları" (Waste - Scrap - Stored Material Fires) and "Bitki Örtüsü Tarımsal Alan Yangınları".

4.3. Feature Importance  
The most influential features for predicting fire type included:

- YAPI_SEKLI (Building Structural Type)

- YANGIN_SEBEBI (Fire Cause)

- ILCE (District)

- KULLANILAN_SU_MIKTARI (Amount of Water Used)

- EKIPLERIN_CIKIS_SAATI (Team Departure Time)

- ADRES_BOLGESI (Address Region)

---

5. Code Description  
The accompanying Jupyter Notebook (erdikilic_project_deliverable.ipynb) implements the entire analysis pipeline:

- Importing Libraries: Loads necessary Python libraries for data manipulation, visualization, statistical analysis, and machine learning.

- Data Loading and Initial Cleaning: Reads the raw CSV data and performs initial cleaning steps like dropping unnecessary columns and converting time strings to minutes.

- Feature Engineering: Creates new features (e.g., total casualties) and maps YANGIN_TURU to broader categories.

- Handling Missing Values: Imputes or removes missing data.

- Exploratory Data Analysis (Implicit): Value counts and descriptive statistics are part of the preprocessing.

- Feature Importance Pre-analysis: A preliminary CatBoost model is trained to visualize feature importances.

- Data Splitting: Divides the data into training, validation, and test sets.

- Preprocessing with ColumnTransformer: Applies StandardScaler to numerical features and OneHotEncoder to categorical features.

- Baseline Model Training and Evaluation: Trains and evaluates the DummyClassifier.

- CatBoost Model Hyperparameter Tuning: Uses RandomizedSearchCV to find optimal hyperparameters for the CatBoost model on the validation set.

- Final CatBoost Model Training and Evaluation: Trains the CatBoost model with the best parameters on the full training set and evaluates it on the held-out test set.

- Results Visualization: Generates confusion matrices and ROC curves for model evaluation.

- Statistical Tests: Performs Shapiro-Wilk tests for normality, Levene's test for homogeneity of variances, Welch's ANOVA for comparing means, and Games-Howell post-hoc tests for pairwise comparisons of response times and water usage across fire types.

---

6. Installation and Requirements  
The analysis is performed using Python. The following libraries are required:

- numpy

- pandas

- matplotlib

- seaborn

- scikit-learn (for train_test_split, RandomizedSearchCV, ColumnTransformer, StandardScaler, OneHotEncoder, DummyClassifier, and various metrics)

- scipy (for statistical functions, likely used by pingouin)

- pingouin (for statistical tests like ANOVA, Welch's ANOVA, Games-Howell)

- shap (mentioned in imports, though not explicitly used in the provided notebook snippet for final results, it's for model interpretability)

- catboost (for the CatBoostClassifier)

These libraries can typically be installed using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy pingouin shap catboost
```

---

7. Usage  
Ensure all required libraries are installed.

Place the dataset 2023_yili_yangin_mudahale_istatistigi.csv in a ../data/ directory relative to the notebook, or update the file path in the notebook.

Run the Jupyter Notebook erdikilic_project_deliverable.ipynb cell by cell.

The notebook will perform data preprocessing, train and evaluate models, conduct statistical tests, and generate output figures (which are also saved to ../reports/figures/).

The cleaned dataset will be saved as ../data/2023_yili_yangin_mudahale_istatistigi_cleaned.csv.

---

8. Data and Code Availability  
The source code, experimental notebooks, and processed data are available at:  
github.com/erdikilic/D1501_Term_Project

---

9. Disclaimer  
This work has benefited from the use of Large Language Models (LLMs), including Gemini, DeepSeek-R1, and DeepL, which were employed to assist in improving the clarity of language, the quality of written expression, and the accuracy of LaTeX formatting. These tools were used exclusively for linguistic and presentational enhancement and did not influence the content or originality of the work.

---

10. References (from the paper)  
[1] C.-Y. Ku and C.-Y. Liu, "Predictive modeling of fire incidence using deep neural networks," Fire, vol. 7. no. 4, p. 136. Apr. 2024. doi: 10.3390/fire7040136.  
[2] Y. Yuan and A. G. Wylie, "Comparing machine learning and time series approaches in predictive modeling of urban fire incidents: A case study of austin, Texas," ISPRS International Journal of Geo-Information, vol. 13, no. 5. p. 149, Apr. 2024. doi: 10.3390/ijgi13050149.  
[3] A. Schmidt, E. Gemmil, and R. Hoskins, Machine learning based risk analysis and predictive modeling of structure fire related casualties, 2024. doi:10.2139/ssrn.4757205.  
[4] G. V. Kuznetsov, A. O. Zhdanova, R. S. Volkov, and P. A. Strizhak, "Optimizing firefighting agent consumption and fire suppression time in buildings by forming a fire feedback loop," Process Safety and Environmental Protection, vol. 165, pp. 754-775. Sep. 2022. doi:10.1016/j.psep.2022.07.061.