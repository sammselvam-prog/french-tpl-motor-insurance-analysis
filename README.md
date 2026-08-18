# French Motor Insurance Claims Data Analysis

## Overview
This project provides an end-to-end data analysis and machine learning pipeline applied to the French Motor Third Party Liability Insurance dataset (freMTPL2freq and freMTPL2sev). The primary objectives are to predict claim occurrences using binary classification models (Logistic Regression and Random Forest), fine-tune hyperparameters via RandomizedSearchCV, extract key features using Recursive Feature Elimination (RFE), and generate actionable business insights using Tableau interactive dashboards.

## Dataset Description
The project utilizes two core datasets merged on the policy identifier (Idpol):

* **freMTPL2freq** : Contains historical policyholder characteristics including policy exposure, vehicle age, driver age, vehicle power, bonus-malus rating, brand, fuel type, area, population density, and region.
* **freMTPL2sev** : Contains the financial claim amounts for policies filed during the target period.

## Pre-Processing & Feature Engineering:

**Data Cleaning & Alignment**: Aggregated severity claims by ```Idpol``` and merged them with policy risk data. Removed 6 mismatched policy IDs present in severity data but absent in frequency data, accounting for ~1.30% (€788,714.18) of total claim amounts.

**Feature Engineering:**

* ```Freq```: Calculated as $\text{ClaimNb} / \text{Exposure}$ to represent claim frequency.

* ```ClaimMade```: Binary target variable assigned 1 if ```ClaimAmount``` $> 0$, otherwise 0.

* ```AgeCategory```: Categorized driver age (```DrivAge```) into Young Adults (18–34), Middle-Aged Adults (35–59), and Seniors (60+).

* ```BonusMalusCategory```: Grouped bonus-malus scores into risk tiers (Excellent, Good, Average, Poor, Very Poor)

**Categorical Encoding**: Converted categorical variables (```VehBrand```, ```VehGas```, ```Area```, ```Region```) using ```LabelEncoder```.

## Visualizations & Exploratory Data Analysis

**Correlation Heatmap**

<img width="558" height="482" alt="CorrHeatmap" src="https://github.com/user-attachments/assets/4091d334-93b5-4f48-b93c-0ed42a0cb0c7" />

> **Description:** Illustrates linear relationships between numerical parameters. Shows a weak positive correlation between ClaimNb and Exposure (0.069), and a strong inverse correlation between driver age (DrivAge) and BonusMalus (-0.480)

**Age Category Distribution**

<img width="520" height="387" alt="Distribution of Age Categories" src="https://github.com/user-attachments/assets/c4c82026-12c9-47e1-8ef6-6e305ba9e92b" />

> **Description:** Bar plot highlighting policyholder volumes across age brackets. Middle-aged adults constitute the majority (~400,000 policies), followed by young adults (~175,000), and seniors (~110,000).

**Pair Plot of Numerical Feature**

<img width="709" height="736" alt="pairplot" src="https://github.com/user-attachments/assets/0568a94b-7456-4718-b5e2-d8b2083700c3" />

> **Description:** Multi-variable scatter matrix demonstrating pairwise distributions among ClaimAmount, Exposure, VehPower, and DrivAge. Shows a heavy concentration of zero claim amounts with severe positive skewness from rare million-euro claims.

**Mean Claim Frequency by Driver Age**

<img width="372" height="262" alt="Freq distribution DrivAge" src="https://github.com/user-attachments/assets/b4744cab-ceee-4c73-9cd3-f63e593fad43" />

> **Description:** Scatter plot mapping average claim frequency against driver age. Highlights elevated frequencies among young drivers (~0.7 around age 20) and elderly drivers (~0.8 at age 95), while maintaining a stable lower baseline (0.2–0.3) for ages 30–80.

**Tableau Visualization: Regional Claim Occurrence**

<img width="1229" height="834" alt="Number of claims based on Regions" src="https://github.com/user-attachments/assets/b8f1a723-1929-4573-9337-b2af015f2dc7" />

> **Description:** Geographic heatmap displaying regional claim density across France. The Centre region displays the highest concentration of total claims (highlighted in deep red), whereas regions like Corse and Bretagne show significantly lower counts.


**Tableau Visualization: Bonus/Malus Distribution by Region**

<img width="1418" height="787" alt="BonusMalus based on Regions" src="https://github.com/user-attachments/assets/4f86d5b8-4280-4118-8154-166fdad8e0f4" />

> **Description:** Regional bar breakdown contrasting Bonus vs. Malus policyholder counts. Centre records the highest overall volume (1,675 Bonus and 154,654 Malus entries), illustrating that high-claim regions contain a elevated share of penalty-rated drivers.

**Tableau Visualization: Average Claim Frequency by Region**

<img width="1229" height="865" alt="Average Frequency on regions" src="https://github.com/user-attachments/assets/6d6073d9-4fd0-4222-a99d-8039aad8a441" />

**Description:** Regional choropleth map shaded by claim frequency severity. Champagne-Ardenne records the highest average claim frequency (0.5813), followed by Franche-Comté (0.4406) and Île-de-France (0.4131), while Haute-Normandie registers the lowest.

## Machine Learning Modeling & Evaluation

Models were trained on an 80/20 train/test split. Performance was evaluated across baseline (default) setups and hyperparameter-tuned setups optimized using ```RandomizedSearchCV```.

| Model | Setup | Accuracy | F1 Score | Precision | Recall | ROC AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | Default | 0.9618 | 0.0193 | 0.2849 | 0.0100 | 0.5045 |
| **Random Forest Classifier** | Default | 0.9903 | 0.8749 | 0.8742 | 0.9045 | 0.9491 |
| **Logistic Regression** | Tuned (`C=0.0001`, `penalty='l1'`, `solver='saga'`) | 0.9619 | 0.0092 | 0.2243 | 0.0047 | 0.5020 |
| **Random Forest Classifier** | Tuned (`n_estimators=178`, `max_depth=110`, `min_samples_split=5`, `min_samples_leaf=2`) | **0.9906** | **0.8801** | **0.8431** | **0.9206** | **0.9570** |


**ROC Comparision**

<img width="578" height="453" alt="ROC curve" src="https://github.com/user-attachments/assets/3d835145-41bf-47be-9988-5bfd241dbdbd" />

>**Description:** ROC curve plot illustrating true positive vs. false positive trade-offs. The Tuned Random Forest model curve hugs the upper-left boundary with an AUC of 0.9570, vastly outperforming Logistic Regression (AUC 0.5020).


**Feature Importance & Selection (RFE)**

<img width="405" height="278" alt="ElbowRFE" src="https://github.com/user-attachments/assets/641086bb-eeee-452e-8b97-9b0bf1845321" />

> **Description:** Elbow plot generated via Recursive Feature Elimination (RFE) displaying accuracy progression against feature count. Demonstrates that top predictive power is achieved using an optimal subset of 3 features (```Freq```, ```VehGas```, ```VehPower```), beyond which performance gains plateau.

## Key Findings & Business Recommendations

* **Top Risk Indicators:** Engine power (```VehPower```), fuel type (```VehGas```), and historical policy claim frequency (```Freq```) represent the key predictors of claim occurrence.

* **Geographical Pricing Strategy:** High-density and high-frequency regions such as *Champagne-Ardenne* and *Île-de-France* warrant localized risk-adjusted premium increases.

* **Demographic Tailoring:** Young drivers (<25) and elderly drivers (>80) demonstrate statistically higher claim frequencies, supporting age-bracketed risk surcharges or telematics-based monitoring programs.

### Built with

Python(Jupyter), Tableau

### Author

Sam Rose M
