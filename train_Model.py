"""C02 Loan Approval Classification

เปรียบเทียบ KNN, SVM Linear, SVM RBF, Decision Tree และ Naive Bayes
เลือก Best Algorithm จาก Mean CV F1 บน Training Set
และใช้ Test Set ประเมินผลหลังจากเลือกโมเดลแล้ว
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import matplotlib

# ใช้ Backend แบบไม่เปิดหน้าต่างกราฟ เพื่อให้รันได้ใน PowerShell
# และ Environment ที่ไม่มี Tcl/Tk เช่น .venv บางรูปแบบ
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# STEP 1: Import Library และกำหนดค่าที่ใช้ในโปรเจกต์
DATA_FILE = "C02_loan_approval_classification.xlsx - Raw_Data.csv"
RANDOM_STATE = 42 #กำหนด Random State เพื่อให้ผลลัพธ์ซ้ำได้
TEST_SIZE = 0.20 #แบ่งข้อมูล 20% เป็น Test Set
CV_FOLDS = 5 #กำหนด Cross-validation จำนวน 5 folds
SCORING = "f1" #ใช้ F1 Score เป็นเกณฑ์ประเมินผล

# เก็บไฟล์ผลลัพธ์ทั้งหมดไว้ในโฟลเดอร์ artifacts
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

NUMERIC_FEATURES = [
    "income_monthly",
    "debt_monthly",
    "credit_score",
    "employment_years",
    "loan_amount",
]
CATEGORICAL_FEATURES = ["home_status"]
HOME_STATUS_MAP = {
    "family": "Family",
    "own": "Own",
    "rent": "Rent",
}


def normalize_home_status(value):
    if pd.isna(value):
        return value

    normalized_value = str(value).strip().casefold()
    if normalized_value not in HOME_STATUS_MAP:
        raise ValueError(
            f"พบค่า home_status ที่ไม่รองรับหลังตัดช่องว่างและปรับตัวพิมพ์: "
            f"{value!r}"
        )
    return HOME_STATUS_MAP[normalized_value]


# STEP 2: Load Dataset
df = pd.read_csv(DATA_FILE)

print("ข้อมูล 5 แถวแรก")
print(df.head())


# STEP 3: Data Understanding
print("\nShape ของข้อมูล =", df.shape) #ดูจำนวนแถวและคอลัมน์ของ DataFrame

print("\nโครงสร้างข้อมูล")
df.info() #ดูข้อมูลแต่ละคอลัมน์ว่ามีชนิดข้อมูลอะไรบ้าง และมี Missing Values หรือไม่

print("\nจำนวน Missing Values")
print(df.isnull().sum()) #ดูจำนวน Missing Values ของแต่ละคอลัมน์

print("\nจำนวนข้อมูลซ้ำ =", df.duplicated().sum()) #ดูจำนวนข้อมูลซ้ำ

print("\nการกระจายของ Target (approved)") #ดูการกระจายของ Target Variable
print(df["approved"].value_counts()) # แสดงจำนวนข้อมูลในแต่ละคลาส
print(df["approved"].value_counts(normalize=True) * 100) # แสดงเปอร์เซ็นต์ของข้อมูลในแต่ละคลาส


# STEP 4: Data Cleaning
# ทำมาตรฐาน home_status ก่อนลบข้อมูลซ้ำ เพื่อให้ค่าที่หมายถึงสถานะเดียวกัน
# เช่น Own, OWN และ " Own " ถูกนับเป็นค่าเดียวกัน
df["home_status"] = df["home_status"].map(normalize_home_status)

print("\nค่า home_status หลังทำมาตรฐาน")
print(df["home_status"].value_counts(dropna=False))

# ลบแถวที่ซ้ำกันทั้งแถวหลังทำมาตรฐานข้อมูลแล้ว
df = df.drop_duplicates().copy()

print("\nShape หลังลบข้อมูลซ้ำ =", df.shape)
print("จำนวนข้อมูลซ้ำหลัง Cleaning =", df.duplicated().sum())

print("\nMissing Values ที่ต้องจัดการ")
print(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].isnull().sum())
print(
    "Numeric จะเติมด้วย Median และ Categorical จะเติมด้วย Mode "
    "ภายใน Pipeline ก่อนฝึกโมเดล"
)


# STEP 5: Check Outliers
# ตรวจสอบ Outlier ด้วย IQR Method แต่ยังไม่ลบหรือแก้ไขค่า
outlier_summary = []

for feature in NUMERIC_FEATURES:
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_count = (
        (df[feature] < lower_bound) | (df[feature] > upper_bound)
    ).sum()

    outlier_summary.append(
        {
            "Feature": feature,
            "Outlier Count": int(outlier_count),
        }
    )

outlier_summary_df = pd.DataFrame(outlier_summary)

print("\nผลการตรวจสอบ Outlier ด้วย IQR Method")
print(outlier_summary_df.to_string(index=False))

features_with_outliers = outlier_summary_df.loc[
    outlier_summary_df["Outlier Count"] > 0, "Feature"
].tolist()

print("\nFeature ที่พบ Outlier =", features_with_outliers)

# สร้าง Box Plot เพื่อดูการกระจายของ Numerical Features
plt.figure(figsize=(12, 6))
df[NUMERIC_FEATURES].boxplot()
plt.title("Box Plot of Numerical Features")
plt.ylabel("Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("boxplot_outliers.png", dpi=300)
plt.close()

print("บันทึก Box Plot แล้วที่ boxplot_outliers.png")


# สร้าง Histogram เพื่อดูการกระจายของแต่ละ Numerical Feature
figure, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.ravel()

for axis, feature in zip(axes, NUMERIC_FEATURES):
    axis.hist(df[feature].dropna(), bins=20, color="#2f80ed", edgecolor="white")
    axis.set_title(f"Distribution of {feature}")
    axis.set_xlabel(feature)
    axis.set_ylabel("Frequency")
    axis.grid(axis="y", alpha=0.25)

axes[-1].axis("off")
figure.suptitle("Histograms of Numerical Features", fontsize=16)
figure.tight_layout()
figure.savefig("histograms_features.png", dpi=300)
plt.close(figure)
print("บันทึก Histogram แล้วที่ histograms_features.png")


# สร้าง Correlation Heatmap ระหว่าง Numerical Features
correlation = df[NUMERIC_FEATURES].corr()
figure, axis = plt.subplots(figsize=(10, 8))
heatmap = axis.imshow(
    correlation,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
)

axis.set_xticks(range(len(NUMERIC_FEATURES)))
axis.set_yticks(range(len(NUMERIC_FEATURES)))
axis.set_xticklabels(NUMERIC_FEATURES, rotation=45, ha="right")
axis.set_yticklabels(NUMERIC_FEATURES)
axis.set_title("Correlation Heatmap of Numerical Features")

for row in range(len(NUMERIC_FEATURES)):
    for column in range(len(NUMERIC_FEATURES)):
        axis.text(
            column,
            row,
            f"{correlation.iloc[row, column]:.2f}",
            ha="center",
            va="center",
            color="black",
        )

figure.colorbar(heatmap, ax=axis, label="Correlation")
figure.tight_layout()
figure.savefig("correlation_heatmap.png", dpi=300)
plt.close(figure)
print("บันทึก Correlation Heatmap แล้วที่ correlation_heatmap.png")


# สร้าง Box Plot แยก Numerical Features ตามค่า Target (approved)
target_values = sorted(df["approved"].dropna().unique())
figure, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.ravel()

for axis, feature in zip(axes, NUMERIC_FEATURES):
    grouped_values = [
        df.loc[df["approved"] == target_value, feature].dropna()
        for target_value in target_values
    ]
    axis.boxplot(grouped_values)
    axis.set_xticks(range(1, len(target_values) + 1))
    axis.set_xticklabels([str(value) for value in target_values])
    axis.set_title(f"{feature} by approved")
    axis.set_xlabel("approved")
    axis.set_ylabel(feature)
    axis.grid(axis="y", alpha=0.25)

axes[-1].axis("off")
figure.suptitle("Numerical Features by Loan Approval", fontsize=16)
figure.tight_layout()
figure.savefig("boxplot_features_by_approved.png", dpi=300)
plt.close(figure)
print(
    "บันทึก Box Plot แยกตาม approved แล้วที่ "
    "boxplot_features_by_approved.png"
)


# STEP 6: Remove Unnecessary Feature
# applicant_id เป็นเพียงรหัสผู้สมัคร ไม่ใช้ทำนายการอนุมัติสินเชื่อ
df = df.drop(columns=["applicant_id"])


# STEP 7: เตรียมการจัดการ Missing Values และ Encoding
# ทุกขั้นตอนอยู่ใน Pipeline เพื่อป้องกัน Data Leakage ระหว่าง Cross-validation
numeric_cleaning = Pipeline(
    [
        ("fill_missing", SimpleImputer(strategy="median")),
    ]
)

categorical_cleaning = Pipeline(
    [
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        ("one_hot_encoding", OneHotEncoder(handle_unknown="ignore")),
    ]
)


# STEP 8: แยก X และ y
X = df.drop(columns=["approved"])
y = df["approved"]

print("\nX shape =", X.shape)
print("y shape =", y.shape)


# STEP 9: Train/Test Split
# แบ่งข้อมูลก่อนทำ GridSearchCV และเก็บ Test Set ไว้ประเมินหลังเลือกโมเดล
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nX_train =", X_train.shape)
print("X_test  =", X_test.shape)
print("y_train =", y_train.shape)
print("y_test  =", y_test.shape)


# STEP 10: เตรียม Preprocessor
# KNN และ SVM ใช้ Scaling
# Decision Tree และ Naive Bayes ไม่จำเป็นต้อง Scaling
scaled_preprocessor = ColumnTransformer(
    [
        (
            "numeric",
            Pipeline(
                [
                    ("fill_missing", SimpleImputer(strategy="median")),
                    ("scaling", StandardScaler()),
                ]
            ),
            NUMERIC_FEATURES,
        ),
        (
            "categorical",
            categorical_cleaning,
            CATEGORICAL_FEATURES,
        ),
    ]
)

unscaled_preprocessor = ColumnTransformer(
    [
        ("numeric", numeric_cleaning, NUMERIC_FEATURES),
        (
            "categorical",
            categorical_cleaning,
            CATEGORICAL_FEATURES,
        ),
    ]
)


# STEP 11: เตรียม Pipeline
# รวม Preprocessing และ Model ไว้ด้วยกัน


# STEP 12: KNN + GridSearchCV
knn_pipeline = Pipeline(
    [
        ("preprocessing", scaled_preprocessor),
        ("model", KNeighborsClassifier()),
    ]
)

knn_parameters = {
    "model__n_neighbors": [3, 5, 7, 9, 11],
    "model__weights": ["uniform", "distance"],
    "model__metric": ["euclidean", "manhattan"],
}

knn_grid = GridSearchCV(
    knn_pipeline,
    knn_parameters,
    scoring=SCORING,
    cv=CV_FOLDS,
    n_jobs=-1,
)
knn_grid.fit(X_train, y_train)


# STEP 13: SVM Linear + GridSearchCV
# ทดลองทั้งไม่ถ่วงน้ำหนักและถ่วงน้ำหนัก Class ด้วย balanced
svm_linear_pipeline = Pipeline(
    [
        ("preprocessing", scaled_preprocessor),
        ("model", SVC(kernel="linear")),
    ]
)

svm_linear_parameters = {
    "model__C": [0.1, 1, 10, 100],
    "model__class_weight": [None, "balanced"],
}

svm_linear_grid = GridSearchCV(
    svm_linear_pipeline,
    svm_linear_parameters,
    scoring=SCORING,
    cv=CV_FOLDS,
    n_jobs=-1,
)
svm_linear_grid.fit(X_train, y_train)


# STEP 14: SVM RBF + GridSearchCV
# C ควบคุมการลงโทษเมื่อทำนายผิด
# gamma ควบคุมขอบเขตอิทธิพลของข้อมูลแต่ละจุด
svm_rbf_pipeline = Pipeline(
    [
        ("preprocessing", scaled_preprocessor),
        ("model", SVC(kernel="rbf")),
    ]
)

svm_rbf_parameters = {
    "model__C": [0.1, 1, 10, 100],
    "model__gamma": ["scale", 0.01, 0.1, 1],
    "model__class_weight": [None, "balanced"],
}

svm_rbf_grid = GridSearchCV(
    svm_rbf_pipeline,
    svm_rbf_parameters,
    scoring=SCORING,
    cv=CV_FOLDS,
    n_jobs=-1,
)
svm_rbf_grid.fit(X_train, y_train)


# STEP 15: Decision Tree + GridSearchCV
# ทดลองทั้งไม่ถ่วงน้ำหนักและถ่วงน้ำหนัก Class ด้วย balanced
decision_tree_pipeline = Pipeline(
    [
        ("preprocessing", unscaled_preprocessor),
        ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ]
)

decision_tree_parameters = {
    "model__criterion": ["gini", "entropy"],
    "model__max_depth": [None, 3, 5, 7, 10],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__class_weight": [None, "balanced"],
}

decision_tree_grid = GridSearchCV(
    decision_tree_pipeline,
    decision_tree_parameters,
    scoring=SCORING,
    cv=CV_FOLDS,
    n_jobs=-1,
)
decision_tree_grid.fit(X_train, y_train)


# STEP 16: Naive Bayes + GridSearchCV
naive_bayes_pipeline = Pipeline(
    [
        ("preprocessing", unscaled_preprocessor),
        ("model", GaussianNB()),
    ]
)

naive_bayes_parameters = {
    "model__var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5],
}

naive_bayes_grid = GridSearchCV(
    naive_bayes_pipeline,
    naive_bayes_parameters,
    scoring=SCORING,
    cv=CV_FOLDS,
    n_jobs=-1,
)
naive_bayes_grid.fit(X_train, y_train)


# STEP 17: รวมโมเดลและเลือก Best Algorithm จาก Mean CV F1
models = {
    "KNN": knn_grid,
    "SVM Linear": svm_linear_grid,
    "SVM RBF": svm_rbf_grid,
    "Decision Tree": decision_tree_grid,
    "Naive Bayes": naive_bayes_grid,
}

cv_results = []

for model_name, grid in models.items():
    cv_results.append(
        {
            "Algorithm": model_name,
            "Best CV F1": grid.best_score_,
            "CV F1 Std": grid.cv_results_["std_test_score"][
                grid.best_index_
            ],
            "Best Parameters": json.dumps(
                grid.best_params_,
                ensure_ascii=False,
                default=str,
            ),
        }
    )

cv_results_df = pd.DataFrame(cv_results)

# ใช้ Mean CV F1 เป็นเกณฑ์เลือกโมเดล ไม่ใช้ Test F1
cv_results_df = cv_results_df.sort_values(
    by="Best CV F1",
    ascending=False,
).reset_index(drop=True)

best_model_name = cv_results_df.loc[0, "Algorithm"]
best_grid = models[best_model_name]
best_model = best_grid.best_estimator_

print("\nผล Cross-validation บน Training Set")
print(cv_results_df.to_string(index=False))

print("\nBest Algorithm จาก Mean CV F1 =", best_model_name)
print("Best CV F1 =", best_grid.best_score_)
print("Best Parameters =", best_grid.best_params_)


# STEP 18: บันทึกโมเดลที่ดีที่สุดของแต่ละ Algorithm และ Best Model
artifact_filenames = {
    "KNN": "knn_best_model.joblib",
    "SVM Linear": "svm_linear_best_model.joblib",
    "SVM RBF": "svm_rbf_best_model.joblib",
    "Decision Tree": "decision_tree_best_model.joblib",
    "Naive Bayes": "naive_bayes_best_model.joblib",
}

for model_name, grid in models.items():
    joblib.dump(
        grid.best_estimator_,
        ARTIFACTS_DIR / artifact_filenames[model_name],
    )

joblib.dump(best_model, ARTIFACTS_DIR / "best_model.joblib")


# STEP 19: ประเมินผลและเปรียบเทียบทั้ง 5 Algorithms
# การเลือก Best Algorithm เสร็จสิ้นแล้วใน STEP 17
# Test Metrics ใช้รายงานผล ไม่ใช้จัดอันดับหรือเปลี่ยน Best Model
results = []

for model_name, grid in models.items():
    y_pred = grid.best_estimator_.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append(
        {
            "Algorithm": model_name,
            "Best CV F1": grid.best_score_,
            "Test Accuracy": accuracy,
            "Test Precision": precision,
            "Test Recall": recall,
            "Test F1": f1,
            "Best Parameters": json.dumps(
                grid.best_params_,
                ensure_ascii=False,
                default=str,
            ),
        }
    )

    print(f"\n{model_name}")
    print("Best CV F1 =", grid.best_score_)
    print("Test Accuracy =", accuracy)
    print("Test Precision =", precision)
    print("Test Recall =", recall)
    print("Test F1 =", f1)
    print("Confusion Matrix (แถว = ค่าจริง, คอลัมน์ = ค่าทำนาย)")
    print(confusion_matrix(y_test, y_pred))

results_df = pd.DataFrame(results)

# จัดเรียงด้วย Best CV F1 เท่านั้น
results_df = results_df.sort_values(
    by="Best CV F1",
    ascending=False,
).reset_index(drop=True)

print("\nตารางเปรียบเทียบโมเดล (จัดอันดับด้วย Best CV F1)")
print(results_df.to_string(index=False))

results_df.to_csv(
    ARTIFACTS_DIR / "model_comparison.csv",
    index=False,
)


# STEP 20: รายงานผลสุดท้ายของ Best Model
best_prediction = best_model.predict(X_test)

best_test_metrics = {
    "test_accuracy": accuracy_score(y_test, best_prediction),
    "test_precision": precision_score(
        y_test, best_prediction, zero_division=0
    ),
    "test_recall": recall_score(
        y_test, best_prediction, zero_division=0
    ),
    "test_f1": f1_score(
        y_test, best_prediction, zero_division=0
    ),
}

print("\nผลประเมินสุดท้ายของ Best Model:", best_model_name)
for metric_name, metric_value in best_test_metrics.items():
    print(f"{metric_name} = {metric_value:.4f}")

print("Confusion Matrix ของ Best Model")
print(confusion_matrix(y_test, best_prediction))


# STEP 21: บันทึก Metadata ของ Best Model
metadata = {
    "best_algorithm": best_model_name,
    "selection_metric": "mean_cv_f1",
    "cv_scoring": SCORING,
    "cv_folds": CV_FOLDS,
    "best_cv_f1": float(best_grid.best_score_),
    "best_parameters": best_grid.best_params_,
    "test_metrics": {
        key: float(value)
        for key, value in best_test_metrics.items()
    },
    "dataset_file": DATA_FILE,
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "scikit_learn_version": sklearn.__version__,
    "class_distribution_after_cleaning": {
        str(key): int(value)
        for key, value in y.value_counts().sort_index().items()
    },
    "artifact_file": "best_model.joblib",
}

with open(
    ARTIFACTS_DIR / "best_model_metadata.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(metadata, file, ensure_ascii=False, indent=4)

print("\nบันทึก Artifacts ใหม่เรียบร้อย")
print("โฟลเดอร์:", ARTIFACTS_DIR.resolve())
print("scikit-learn version =", sklearn.__version__)
