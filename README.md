# Data_Prep: Exploratory Analysis, Feature Engineering, Feature Selection, Data Cleaning/Imputation
This project demonstrates the workflow for preparing data for a classification task. I begin with exploratory analysis, then perform feature engineering to extract additional features from the raw data. Next, I conduct preliminary data cleaning using insights from the exploratory and feature-engineering steps and test various imputation methods. Finally, I demonstrate feature selection on the cleaned data as described below.
## Exploratory Data Analysis

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

1. **Distribution Plots**:

   * Histograms and KDE plots for each numerical feature.
   * Overlaid target classes (hue) to detect differences between classes.

2. **Pairwise Relationships**:

   * PairGrid with KDE contours on lower/upper triangles and KDE on diagonal.
   * Numerical features statistics vs binned features

3. **Feature–Target Mutual Information**:

   * Computed pairwise mutual information among numerical features.
   * Computed mutual information between each numerical feature and the binary target.
   * Visualized MI matrices as heatmaps to highlight strongest associations.

---

### 4. Categorical Features

1. **Cardinality**:

   * Counted unique levels in each categorical column to identify high-cardinality features.

2. **Gridplot of Histograms**:

   * Custom grid of proportional histograms: each subplot shows distribution of one categorical feature, with hue by another feature (including the target).

3. **Mutual Information**:

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



