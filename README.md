# <span style = "color:purple">Data_Prep: Exploratory Analysis, Feature Engineering, Feature Selection, Data Cleaning/Imputation</span>
This project demonstrates the workflow for preparing data for a classification task. I begin with exploratory analysis, then perform feature engineering to extract additional features from the raw data. Next, I conduct preliminary data cleaning using insights from the exploratory and feature-engineering steps and test various imputation methods. Finally, I demonstrate feature selection on the cleaned data as described below.

## <span style = "color:purple">Exploratory Data Analysis</span>

The goal of exploratory data analysis is to understand data quality, distributions, relationships, and to prepare for robust feature engineering, informed imputation, and feature selection ahead of modeling.

---

### 1. Initial Observations

* **Target Variable**: Here the target is a binary variable indicating the outcome of interest.
* **Feature Types**:
  * **Numerical Features**: Continuous or integer-valued columns (here, 6 numerical features).
  * **Categorical and Binary Features**: Here we have 5 features with discrete labels and 2 binary features.
* **Missing Values**: percent missing values for all data and by feature. Here, all features have comparable number of missing values.

---

### 2. Missing Values & Class Distribution

* **Missing Values Distribution**:

  * Generated a heatmap of nulls to visualize patterns and block structures.
  * Tabulated and plotted a bar chart of percentage missing per feature and per class to prioritize imputation.
  * Here missing data seems to be at random.

* **Class Distribution**:

  * Counted frequency of each class in the target variable to assess balance.
  * Plotted a bar chart highlighting any imbalance (Here class is balanced).

---

### 3. Numerical Features

* **Distribution Plots**:

   * Histograms and KDE plots for each numerical feature.
   * Overlaid target classes (hue) to detect differences between classes.

* **Pairwise Relationships**:

   * PairGrid with KDE contours on lower/upper triangles and KDE on diagonal.
   * Numerical features statistics vs binned features

* **Feature–Target Mutual Information**:

   * Computed pairwise mutual information among numerical features.
   * Computed mutual information between each numerical feature and the binary target.
   * Visualized MI matrices as heatmaps to highlight strongest associations.

---

### 4. Categorical Features

* **Cardinality**:

   * Counted unique levels in each categorical column to identify high-cardinality features.

* **Gridplot of Histograms**:

   * Custom grid of proportional histograms: each subplot shows distribution of one categorical feature, with hue by another feature (including the target).

* **Mutual Information**:

   * Computed MI matrices for categorical variables (treating them as discrete).
   * Computed MI between each categorical feature and the target.
   * Plotted heatmaps to compare strength of relationships.

---

### 5. Observations

* 24% of data has at least one missing value. That's a lot of data to lose if we are to simply delete all rows with null values.
* The percent of missing data from each column is comparable.
* Identified patterns to inform data cleaning (filling in null values). Any values that I am not able to fill using these patterns can be filled later via imputation (see Imputation Methods)
* Identified a possible method to combine or bin some of the numerical features. I will look at these in more depth later (see Feature Engineering)
* Identified some features to possibly dismiss. This will be determined later (see Feature Selection)
* Identified Categorical Features that can be concatenated to create a composite feature.

## <span style = "color:purple">Feature Engineering</span>

The goal of featrure engineering is to explore relationships between features and between features and target variable in order to generate meaningful new features by transforming, combining, and/or encoding existing ones.

---

### 1. Numerical Features

* **Correlation Analysis**: 

   * Plotted pairwise correlations between numerical variables to identify redundancy and strong associations with the target.

* **Distribution Plots**:

   * Used histograms, empirical cumulative distribution functions, violin plots, and pairplots to examine the shape, spread, and overlaps in distributions
   * Analysed the distributions as a function of the target and other binary features with strong correlation with the target.

* **Feature Transformations**:

   * Considered binning continuous variables into intervals based on observed trends
   * Created combined features and transformed some into binary based on observations showing potential signal

* **Feature effect on target Predictability**:

   * Compared the usefulness of the original features and the engineered features on several baseline algorithms (see below)

---

### 2. Categorical Features

* **Low Cardinality Nominal Variables**: 

   * Combined selected features where interactions revealed meaningful separations in the target distribution
   * Applied one-hot encoding

* **Low Cardinality Ordinal Variables**:

   * Transformed selected features into binary variables whenever splits showed potential predictive signal
   * Treated most as numeric features to respect inherent ordering

* **High Cardinality Variables**:

   * Grouped/binned into small number of categories based on observed frequencies and target-driven insights
   * Tested whether binning improved predictive power across models

---

### 3. Model Benchmarking

To measure the usefulness of engineered features, I tested several baseline algorithms:

* **Naive Bayes**
* **Logistic Regression**
* **LightGBM**
* **CatBoost**

Model performance was evaluated using 10-fold or repeated 10-fold cross-validation, using both the mean classification accuracy and standard deviation across folds.

---

## <span style = "color:purple">Data Cleaning and Imputation</span>

The goal of data cleaning and imputation is to resolve inconsistencies and fill in missing values in a principled way, informed by insights from exploratory analysis and feature engineering, or/and domain specific knowledge

---

### 1. Assessing Type of Missing Values

There are different types of missing values. Missing values that are independent of all other variables (observed features and target variable) are called Missing Completely at Random (MCAR). MCAR values can be dropped or imputed safely. Missing values that correlate with observed features but not target variable are called Missing at Random (MAR). For MAR values, we need to use imputation methods informed by related observed features. When missing values are dependent on the value of the missing data itself, Missing Not at Random (MNAR), then we need to model missingness as standard imputation methods would not be suitable.

  * Analyzed patterns of the missing values across all features.
  * Statistical checks confirmed that the data are **Missing Completely at Random (MCAR)**.
  * While this means null rows could be dropped without introducing bias, doing so would reduce the training sample size significantly and leave missing values in the test set unresolved.

---

### 2. Rule-Based Pre-Imputation

Some missing values could be filled deterministically using **logical or domain-driven rules**. Based on observations from exploratory analysis and feature engineering, it was possible to fill in certain missing values. Doing so, however, left remaining MAR values that require imputation as described below.

---

### 3. Imputation Methods

For the remaining missing data, I applied three systematic approaches:

1. **<span style="color:blue">impute\_method = flag</span>**

   * Replaced null values with a flag value lying outside the feasible numeric range for each feature. This provides models with an explicit signal that the value was missing.

2. **<span style="color:blue">impute\_method = Impute</span>**

   * Used context-specific imputations combining:

      * Domain-knowledge rules (here based on observations from exploratory analysis and feature engineering).
      * k-Nearest Neighbor–based imputation for correlated numeric features.
      * Simple statistical imputations (mean/median/mode).

3. **<span style="color:blue">impute\_method = Impute\_flag</span>**

   * Builds on `Impute`, but also creates **binary flag variables** marking which values were imputed.

---

### 3. Model Benchmarking

To compare the imputation methods, I tested the data on several baseline algorithms:

* **Logistic Regression**
* **Random Forest**
* **Support Vector**
* **LightGBM**
* **CatBoost**

Model performance was evaluated using 10-fold or repeated 10-fold cross-validation, using both the mean classification scores and standard deviation across folds. The metric scores used: accuracy, roc_auc, f1, precision, and recall

---