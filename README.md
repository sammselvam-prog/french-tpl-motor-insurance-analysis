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




