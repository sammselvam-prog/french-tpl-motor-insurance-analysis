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


