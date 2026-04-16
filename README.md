# <span style = "color:purple">Tabular Classification Pipeline: Exploratory Analysis, Feature Engineering, Feature Selection, Data Cleaning/Imputation, and Deep Learning Benchmarks</span>
This project demonstrates the workflow for preparing data for a classification task. I begin with exploratory analysis, then perform feature engineering to extract additional features from the raw data. Next, I conduct preliminary data cleaning using insights from the exploratory and feature-engineering steps and test various imputation methods. I then demonstrate feature selection on the cleaned data as described below. Finally, I compare the winning CatBoost baseline against several deep learning architectures (MLP, TabResNet, CatEmb-MLP, FT-Transformer) with optional post-hoc calibration and k-fold ensembles.

---

## Table of Contents

1. [Exploratory Data Analysis](#exploratory-data-analysis)
2. [Feature Engineering](#feature-engineering)
3. [Data Cleaning and Imputation](#data-cleaning-and-imputation)
4. [Feature Selection](#feature-selection)
5. [Hyperparameter Optimization](#hyperparameter-optimization)
6. [Deep Learning Models](#deep-learning-models)
7. [Results](#results)

---

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
  * Computed missingness-indicator Phi coefficients (near-zero correlations) and ran **Little's MCAR test** (χ² = 80.79, df = 82, p = 0.517), confirming the data are **Missing Completely at Random (MCAR)**.

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

The goal of feature engineering is to explore relationships between features and between features and target variable in order to generate meaningful new features by transforming, combining, and/or encoding existing ones.

---

### 1. Numerical Features

* **Correlation Analysis**: 

   * Plotted pairwise correlations between numerical variables to identify redundancy and strong associations with the target.

* **Distribution Plots**:

   * Used histograms, empirical cumulative distribution functions, violin plots, and pairplots to examine the shape, spread, and overlaps in distributions
   * Analyzed the distributions as a function of the target and other binary features with strong correlation with the target.

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

  * Visualized missingness patterns via missingness matrices for both train and test sets, visually confirming the random structure established in EDA.
  * Since the data are **MCAR** (confirmed in EDA), null rows could be dropped without introducing bias, but doing so would reduce the training sample size significantly and leave missing values in the test set unresolved.

---

### 2. Rule-Based Pre-Imputation

Some missing values could be filled deterministically using **logical or domain-driven rules**. Based on observations from exploratory analysis and feature engineering, it was possible to fill in certain missing values. Doing so, however, left remaining MAR values that require imputation as described below.

---

### 3. Imputation Methods

> **Note (Kaggle/tutorial context):** In this project, imputation is applied to train and test data jointly — they are concatenated before any cleaning or imputation runs, then split back afterward. This is valid here because the full test set is available upfront (as is typical in Kaggle competitions), and it allows group- and family-based rules to work across the split boundary. **In a real production setting this would not be appropriate**, since future data arrives after the model is deployed. In that case, all imputation parameters (KNN neighbors, lookup tables, statistical fallbacks) must be fitted on training data only and then applied to new data at inference time — or you should use models that handle missing values natively (e.g. CatBoost, LightGBM, XGBoost).

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
* **Support Vector Machine (SVM)**
* **LightGBM**
* **CatBoost**

Model performance was evaluated using 10-fold or repeated 10-fold cross-validation, using both the mean classification scores and standard deviation across folds. The metric scores used: accuracy, roc_auc, f1, precision, and recall

---

## <span style = "color:purple">Feature Selection</span>

The goal of feature selection is to reduce redundancy in order to control overfitting and thus improve model performance and reduce computational cost. Furthermore, feature selection improves model interpretability as only the most informative predictors are retained. This stage was carried out after data cleaning and imputation to ensure that all candidate features were consistently available. I compare several selection strategies and evaluate them with CatBoostClassifier using a consistent cross-validation protocol.

---

### Methods Evaluated

* Mutual Information (a Filter-Based Method)

   * Compute the mutual information between each feature and the target and rank the features accordingly
   * Evaluate the model performance selecting features with thresholds at ≥30% of the maximum MI, ≥10%, ≥5%, and all features with MI > 0.

* Embedded Feature Importance (CatBoost)

   * Train CatBoostClassifier model and rank features using the get_feature_importance catboost method.
   * Evaluate the model performance selecting features with thresholds at ≥10% of the maximum importance, ≥1%, and all features with importance > 0.

* Lasso Cross-Validation

   * Fit LassoCV with 5-fold cross validation
   * Evaluate the model performance keeping only features with non-zero coefficients, and also threshold by absolute coeffiecient at ≥30%, ≥10%, and ≥5% of the maximum absolute coefficient.

* Permutation Importance

   * Train CatBoostClassifier on the fold’s training partition and compute permutation importance on the fold’s validation split during repeated stratified 5-fold cross-validation. The final PI values are the mean PI across the folds
   * Evaluate thresholds at ≥3% of the maximum PI, ≥1%, and all features with PI > 0.

* Recursive Feature Elimination Cross Validation (a Wrapper-Based Method)

   * Using CatBoostClassifier as estimator, eliminate the least important features
   * Evaluate model performance with the selected subset compared to its performance using all features

* Principal Component Analysis (PCA, Dimensionality Reduction)

   * Replace original features with top-k principal componenets fitted on training data.
   * Evaluate model performance keeping components by 90%, 95%, and 99% explained variance.

---

### Evaluation

* Use Stratified 10-fold cross validation with CatBoostClassifier
* Report the mean ± standard deviation for accuracy, roc_auc, f1, precision, recall on validation folds
* Report each model's prediction accuracy on the held out test set

----

### Hyperparameter Optimization

* Optimize CatBoostClassifier hyperparameters on the winning feature selection uing Optuna
* Report final test evaluation using the best trial parameters

---

## <span style = "color:purple">Deep Learning Models</span>

I compare the performance of the optimized CatBoostClassifier against several neural network models tailored for tabular data.

---

### Models

* **Multi-Layer Perceptron (MLP)**: Fully connected network with batch normalization, dropout, and custom activation functions.
* **TabResNet**: ResNet-style architecture for tabular data with residual blocks, based on *Revisiting Deep Learning Models for Tabular Data* (Gorishniy et al.).
* **Categorical Embedding MLP (CatEmb_MLP)**: MLP with learned embeddings for categorical features, allowing variable-size per-feature embedding dimensions.
* **FT-Transformer**: Feature Tokenizer + Transformer architecture (Gorishniy et al.) with modifications including a parametric mish activation (PMish), GLU variants (ReGLU, GeGLU, MiGLU), linear and periodic continuous feature embeddings, and an MLP classification head.

---

### Training & Evaluation

* **Ensemble cross-validation**: Use repeated Stratified K-Fold and average predictions across all folds.
* **Calibration**: I use an optional post-hoc probability calibration: Temperature Scaling (TS), entropy-aware Heteroscedastic Temperature Scaling (HTS), and logit-aware HnLTS.
* **Parallelism**: This is optional and would depend on available CPUs.
* **Data augmentation**: Optional mixup augmentation with Beta-distributed interpolation of continuous features and label-aware selection for categorical features.
* **Hyperparameter tuning**: Use Optuna optimization for MLP, TabResNet, and CatEmb_MLP architectures.
