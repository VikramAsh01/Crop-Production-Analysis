import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score,accuracy_score,confusion_matrix,classification_report)
sns.set_theme(style="whitegrid")
df = pd.read_csv("D:\\College\\Semester 5\\INT234\\crop_production_uncleaned.txt")
print("\nFIRST 5 ROWS")
print(df.head())
print("\nDATASET SHAPE")
print(df.shape)
print("\nDATASET INFORMATION")
df.info()

#Performing EDA(Exploratory Data Analysis)
df.columns = df.columns.str.strip()
text_columns = ["State_Name", "District_Name", "Season", "Crop"]
for col in text_columns:
    df[col] = df[col].astype("string").str.strip().str.title()
for col in ["Crop_Year", "Area", "Production"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
print("\nUNIQUE CROPS:", df["Crop"].nunique())
print("UNIQUE STATES:", df["State_Name"].nunique())
print("SEASONS:", sorted(df["Season"].dropna().unique()))
print("\nMISSING VALUES BEFORE CLEANING")
print(df.isnull().sum())
print("\nMISSING VALUE PERCENTAGE")
print((df.isnull().mean() * 100).round(2))
print("\nDUPLICATE ROWS:", df.duplicated().sum())
df = df.drop_duplicates()
print("\nNEGATIVE AREA:", (df["Area"] < 0).sum())
print("NEGATIVE PRODUCTION:", (df["Production"] < 0).sum())
print("ZERO AREA:", (df["Area"] == 0).sum())
print("ZERO PRODUCTION:", (df["Production"] == 0).sum())
print(
    "YEAR RANGE:",
    df["Crop_Year"].min(),
    "to",
    df["Crop_Year"].max()
)

# Negative area and production values are invalid.
df.loc[df["Area"] < 0, "Area"] = np.nan
df.loc[df["Production"] < 0, "Production"] = np.nan
print("\nMISSING VALUES BEFORE FILLING")
print(df.isnull().sum())
df["Area"] = df["Area"].fillna(df["Area"].median())
df = df.dropna(subset=["Production"])
df = df.dropna(
    subset=[
        "Crop_Year",
        "State_Name",
        "District_Name",
        "Season",
        "Crop"
    ]
)

print("\nMISSING VALUES AFTER CLEANING")
print(df.isnull().sum())
# 8. STATISTICAL SUMMARY
print("\nSTATISTICAL SUMMARY")
print(df.describe())

print("\nMEDIAN VALUES")
print(df[["Crop_Year", "Area", "Production"]].median())

print("\nSKEWNESS")
print(df[["Area", "Production"]].skew())

# 9. OUTLIER DETECTION
# Use log-transformed boxplots so extreme values do not
# compress the rest of the plot. Outliers are not deleted.

plt.figure(figsize=(8, 4))
sns.boxplot(x=np.log1p(df["Area"]))
plt.title("Crop Area Outliers (Log Transformed)")
plt.xlabel("log(1 + Area)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(x=np.log1p(df["Production"]))
plt.title("Crop Production Outliers (Log Transformed)")
plt.xlabel("log(1 + Production)")
plt.tight_layout()
plt.show()

for col in ["Area", "Production"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower_limit) |
        (df[col] > upper_limit)
    ]

    print("\nOUTLIERS IN", col)
    print("Lower limit:", lower_limit)
    print("Upper limit:", upper_limit)
    print("Number of outliers:", len(outliers))

# 10. IMPROVED DISTRIBUTION HISTOGRAMS
# log1p reduces the visual impact of very large values.
# Zero values can still be included.

plt.figure(figsize=(8, 5))
plt.hist(np.log1p(df["Area"]), bins=50, edgecolor="black")
plt.title("Distribution of Crop Area (Log Transformed)")
plt.xlabel("log(1 + Area)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(np.log1p(df["Production"]), bins=50, edgecolor="black")
plt.title("Distribution of Crop Production (Log Transformed)")
plt.xlabel("log(1 + Production)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 11. TOP 10 CROPS BY TOTAL PRODUCTION
# Logarithmic y-axis makes smaller crop totals easier to see.

top_crops = (
    df.groupby("Crop")["Production"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTOP 10 CROPS BY TOTAL PRODUCTION")
print(top_crops)

plt.figure(figsize=(10, 6))
top_crops.sort_values().plot(kind="barh")
plt.yscale("linear")
plt.xscale("log")
plt.title("Top 10 Crops by Total Production (Log Scale)")
plt.xlabel("Total Production (Log Scale)")
plt.ylabel("Crop")
plt.tight_layout()
plt.show()
# 12. TOTAL PRODUCTION BY YEAR
# A log y-axis makes year-to-year changes easier to inspect.

yearly_production = df.groupby("Crop_Year")["Production"].sum()

print("\nTOTAL PRODUCTION BY YEAR")
print(yearly_production)

plt.figure(figsize=(10, 5))
yearly_production.plot(kind="line", marker="o")
plt.yscale("log")
plt.title("Total Crop Production by Year (Log Scale)")
plt.xlabel("Year")
plt.ylabel("Total Production (Log Scale)")
plt.tight_layout()
plt.show()

# 13. PRODUCTION BY SEASON

season_production = df.groupby("Season")["Production"].sum()
print("\nPRODUCTION BY SEASON")
print(season_production)
plt.figure(figsize=(9, 5))
season_production.sort_values().plot(kind="bar")
plt.yscale("log")
plt.title("Total Production by Season (Log Scale)")
plt.xlabel("Season")
plt.ylabel("Total Production (Log Scale)")
plt.xticks(rotation=35)
plt.tight_layout()
plt.show()

# 14. AREA VS PRODUCTION SCATTER PLOT
# Use only positive values because a log axis cannot show zero.
# A sample keeps the graph readable.

positive_data = df[
    (df["Area"] > 0) &
    (df["Production"] > 0)
]
sample_df = positive_data.sample(
    n=min(5000, len(positive_data)),
    random_state=42
)
plt.figure(figsize=(8, 5))
plt.scatter(
    sample_df["Area"],
    sample_df["Production"],
    alpha=0.3,
    s=15
)
plt.xscale("log")
plt.yscale("log")
plt.title("Area vs Production (Log-Log Scale)")
plt.xlabel("Crop Area (Log Scale)")
plt.ylabel("Crop Production (Log Scale)")
plt.tight_layout()
plt.show()

# 15. CORRELATION

correlation = df[["Crop_Year", "Area", "Production"]].corr()

print("\nCORRELATION MATRIX")
print(correlation)

plt.figure(figsize=(7, 5))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".3f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# 16. NORMALIZATION FOR EDA
# Keep original Area and Crop_Year columns unchanged.

if df["Area"].max() != df["Area"].min():
    df["Area_Normalized"] = (
        (df["Area"] - df["Area"].min()) /
        (df["Area"].max() - df["Area"].min())
    )
else:
    df["Area_Normalized"] = 0

if df["Crop_Year"].max() != df["Crop_Year"].min():
    df["Year_Normalized"] = (
        (df["Crop_Year"] - df["Crop_Year"].min()) /
        (df["Crop_Year"].max() - df["Crop_Year"].min())
    )
else:
    df["Year_Normalized"] = 0

print("\nNORMALIZED DATA")
print(
    df[
        [
            "Area",
            "Area_Normalized",
            "Crop_Year",
            "Year_Normalized"
        ]
    ].head()
)

# 17. SAVE CLEANED DATASET


output_path = "D:\\College\\Semester 5\\INT234\\crop_production_cleaned.csv"
df.to_csv(output_path, index=False)

print("\nCLEANED DATA SAVED TO:", output_path)
print("FINAL DATASET SHAPE:", df.shape)

print("\nFINAL MISSING VALUES")
print(df.isnull().sum())

print("\nFINAL DUPLICATE ROWS")
print(df.duplicated().sum())
# MODEL 1: SIMPLE LINEAR REGRESSION
# Predict Production using Area only.
X1 = df[["Area"]]
y1 = df["Production"]

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42
)

model1 = LinearRegression()
model1.fit(X1_train, y1_train)
pred1 = model1.predict(X1_test)

mae1 = mean_absolute_error(y1_test, pred1)
mse1 = mean_squared_error(y1_test, pred1)
rmse1 = np.sqrt(mse1)
r2_1 = r2_score(y1_test, pred1)

print("\nMODEL 1: SIMPLE LINEAR REGRESSION")
print("MAE:", mae1)
print("MSE:", mse1)
print("RMSE:", rmse1)
print("R2 Score:", r2_1)

plot_data = pd.DataFrame({
    "Area": X1_test["Area"].values,
    "Actual": y1_test.values,
    "Predicted": pred1
}).sort_values("Area")

plt.figure(figsize=(8, 5))
plt.scatter(
    plot_data["Area"],
    plot_data["Actual"],
    alpha=0.25,
    s=12,
    label="Actual"
)
plt.plot(
    plot_data["Area"],
    plot_data["Predicted"],
    color="red",
    linewidth=2,
    label="Predicted"
)
plt.xscale("log")
plt.yscale("symlog")
plt.xlabel("Area (Log Scale)")
plt.ylabel("Production (Symmetric Log Scale)")
plt.title("Simple Linear Regression: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.show()

# MODEL 2: MULTIPLE LINEAR REGRESSION
# Predict Production using Area, Year, Season and Crop.


X2 = df[["Area", "Crop_Year", "Season", "Crop"]]
X2 = pd.get_dummies(
    X2,
    columns=["Season", "Crop"],
    dtype=int
)
y2 = df["Production"]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

model2 = LinearRegression()
model2.fit(X2_train, y2_train)
pred2 = model2.predict(X2_test)

mae2 = mean_absolute_error(y2_test, pred2)
mse2 = mean_squared_error(y2_test, pred2)
rmse2 = np.sqrt(mse2)
r2_2 = r2_score(y2_test, pred2)

print("\nMODEL 2: MULTIPLE LINEAR REGRESSION")
print("MAE:", mae2)
print("MSE:", mse2)
print("RMSE:", rmse2)
print("R2 Score:", r2_2)


# MODEL 3: POLYNOMIAL REGRESSION
# Degree 2 polynomial using Area.

X3 = df[["Area"]]
y3 = df["Production"]

X3_train, X3_test, y3_train, y3_test = train_test_split(
    X3, y3, test_size=0.2, random_state=42
)

poly = PolynomialFeatures(degree=2)
X3_train_poly = poly.fit_transform(X3_train)
X3_test_poly = poly.transform(X3_test)

model3 = LinearRegression()
model3.fit(X3_train_poly, y3_train)
pred3 = model3.predict(X3_test_poly)

mae3 = mean_absolute_error(y3_test, pred3)
mse3 = mean_squared_error(y3_test, pred3)
rmse3 = np.sqrt(mse3)
r2_3 = r2_score(y3_test, pred3)

print("\nMODEL 3: POLYNOMIAL REGRESSION")
print("MAE:", mae3)
print("MSE:", mse3)
print("RMSE:", rmse3)
print("R2 Score:", r2_3)

# PREPARE CLASSIFICATION DATA
# Production is divided into Low, Medium and High categories.


production_rank = df["Production"].rank(method="first")

df["Production_Category"] = pd.qcut(
    production_rank,
    q=3,
    labels=["Low", "Medium", "High"]
)

print("\nPRODUCTION CATEGORY COUNTS")
print(df["Production_Category"].value_counts())

# Use a sample to keep KNN and SVM computationally manageable.
classification_df = df.sample(
    n=min(20000, len(df)),
    random_state=42
)

X_class = classification_df[
    ["Area", "Crop_Year", "Season", "Crop"]
]

X_class = pd.get_dummies(
    X_class,
    columns=["Season", "Crop"],
    dtype=int
)

y_class = classification_df["Production_Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X_class,
    y_class,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)

# Scale only after splitting to avoid fitting scaler on test data.
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# MODEL 4: DECISION TREE CLASSIFIER


model4 = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=42
)

model4.fit(X_train, y_train)
pred4 = model4.predict(X_test)
accuracy4 = accuracy_score(y_test, pred4)

print("\nMODEL 4: DECISION TREE CLASSIFIER")
print("Accuracy:", accuracy4)
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred4))
print("Classification Report:")
print(classification_report(y_test, pred4, zero_division=0))

plt.figure(figsize=(16, 8))
plot_tree(
    model4,
    max_depth=3,
    feature_names=list(X_class.columns),
    class_names=[str(c) for c in model4.classes_],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Decision Tree Classifier")
plt.tight_layout()
plt.show()

# MODEL 5: K-NEAREST NEIGHBORS

model5 = KNeighborsClassifier(n_neighbors=5)
model5.fit(X_train_scaled, y_train)
pred5 = model5.predict(X_test_scaled)
accuracy5 = accuracy_score(y_test, pred5)

print("\nMODEL 5: K-NEAREST NEIGHBORS")
print("Accuracy:", accuracy5)
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred5))
print("Classification Report:")
print(classification_report(y_test, pred5, zero_division=0))

# MODEL 6: SUPPORT VECTOR MACHINE

model6 = SVC(kernel="linear")
model6.fit(X_train_scaled, y_train)
pred6 = model6.predict(X_test_scaled)
accuracy6 = accuracy_score(y_test, pred6)

print("\nMODEL 6: SUPPORT VECTOR MACHINE")
print("Accuracy:", accuracy6)
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred6))
print("Classification Report:")
print(classification_report(y_test, pred6, zero_division=0))

# 18. COMPARE REGRESSION MODELS

regression_results = pd.DataFrame({
    "Model": [
        "Simple Linear Regression",
        "Multiple Linear Regression",
        "Polynomial Regression"
    ],
    "MAE": [mae1, mae2, mae3],
    "MSE": [mse1, mse2, mse3],
    "RMSE": [rmse1, rmse2, rmse3],
    "R2 Score": [r2_1, r2_2, r2_3]
})

print("\nREGRESSION MODEL COMPARISON")
print(
    regression_results.sort_values(
        "R2 Score",
        ascending=False
    ).to_string(index=False)
)

# 19. COMPARE CLASSIFICATION MODELS

classification_results = pd.DataFrame({
    "Model": ["Decision Tree", "KNN", "SVM"],
    "Accuracy": [accuracy4, accuracy5, accuracy6]
})

print("\nCLASSIFICATION MODEL COMPARISON")
print(
    classification_results.sort_values(
        "Accuracy",
        ascending=False
    ).to_string(index=False)
)

plt.figure(figsize=(8, 5))
plt.bar(
    classification_results["Model"],
    classification_results["Accuracy"]
)
plt.title("Classification Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# Save model comparison tables too.
regression_results.to_csv("D:\\College\\Semester 5\\INT234\\regression_model_comparison.csv", index=False)
classification_results.to_csv("D:\\College\\Semester 5\\INT234\\classification_model_comparison.csv", index=False)

print("\nALL EDA, CLEANING AND MODELLING STEPS COMPLETED.")
print("Created files:")
print("D:\\College\\Semester 5\\INT234\\crop_production_cleaned.csv")
print("D:\\College\\Semester 5\\INT234\\regression_model_comparison.csv")
print("D:\\College\\Semester 5\\INT234\\classification_model_comparison.csv")
