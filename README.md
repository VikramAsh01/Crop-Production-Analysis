# Indian Crop Production Analysis: EDA & Machine Learning

An end-to-end data analytics and machine learning project exploring historical Indian crop production records from 1997 to 2015.

## Project Overview

This project uses Python to clean agricultural data, perform exploratory data analysis (EDA), identify patterns, and compare regression and classification algorithms.

The objective is to understand crop production patterns across crops, seasons, years, and cultivated areas while evaluating the predictive capabilities of different machine learning models.

## Tech Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Techniques:** Data cleaning, EDA, normalization, outlier detection, regression, classification, model evaluation

## Dataset

- Original records: 246,391
- Cleaned records: 241,763
- Original features: 7
- Study period: 1997–2015

Features include state, district, crop year, season, crop, cultivated area, and production.

*Note: The original dataset publisher and production units require verification.*

## Project Workflow

1. Data inspection and preprocessing
2. Missing-value handling and duplicate removal
3. Outlier detection using the Interquartile Range (IQR) method
4. Exploratory data analysis and visualization
5. Min-max normalization of selected numerical features
6. Regression model training and evaluation
7. Production category classification
8. Comparative model evaluation

## Exploratory Data Analysis

The analysis includes:
- Crop-wise production comparisons
- Year-wise and season-wise production trends
- Area and production distributions
- Outlier analysis using log transformations
- Area versus production scatterplot
- Correlation heatmap

## Machine Learning Models

### Regression
- Simple Linear Regression
- Multiple Linear Regression
- Polynomial Regression (degree 2)

### Classification
- Decision Tree Classifier
- K-Nearest Neighbors (KNN)
- Linear Support Vector Machine (SVM)

## Key Results

| Metric | Result |
|---|---:|
| Cleaned records | 241,763 |
| Best regression model | Multiple Linear Regression |
| Best regression R² | 0.1185 |
| Best classification model | Decision Tree |
| Decision Tree accuracy | 75.23% |
| KNN accuracy | 71.08% |
| Linear SVM accuracy | 54.58% |

## Key Findings

- Crop production values are highly right-skewed and contain substantial outliers.
- The overall linear correlation between cultivated area and production is weak.
- Multiple Linear Regression performs best among the tested regressors, but its explanatory power remains limited.
- The Decision Tree performs best on the derived Low, Medium, and High production classification task.

**Important:** Classification accuracy measures prediction of relative production categories, not continuous production quantities. The regression results indicate that additional features and stronger validation are needed for reliable forecasting.

## Repository Structure

```text
Indian-Crop-Production-Analysis/
├── README.md
├── Project2.py
├── CropProductionAnalysis.pdf
├── requirements.txt
└── images/
    ├── top_10_crops.png
    ├── correlation_heatmap.png
    ├── classification_comparison.png
    └── actual_vs_predicted.png
```

## Limitations and Future Improvements

- Verify dataset provenance and measurement units.
- Add reliable rainfall, temperature, soil, and irrigation data.
- Investigate crop-specific models and log-transformed targets.
- Use time-based or region-grouped evaluation to test generalization.
- Explore ensemble models, hyperparameter tuning, and cross-validation.

## How to Run

1. Clone this repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Download the dataset from its verified source and configure the input path in `Project2.py`.
4. Run `python Project2.py`.

## Author

**Vikram Ash**  
B.Tech Computer Science Engineering — Data Science  
Lovely Professional University

If you find this project useful, consider giving the repository a star.
