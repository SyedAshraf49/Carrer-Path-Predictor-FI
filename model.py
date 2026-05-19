# model.py
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")

CLEAN_DATA_PATH = os.path.join(DATA_DIR, "training_data_clean.csv")
RAW_DATA_PATH = os.path.join(DATA_DIR, "training_data_comprehensive.csv")

data_path = CLEAN_DATA_PATH if os.path.exists(CLEAN_DATA_PATH) else RAW_DATA_PATH
df = pd.read_csv(data_path)

categorical_cols = ["passion", "hobby", "education_level"]
numeric_cols = [
    "age",
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
    "salary_importance",
    "work_life_balance",
]

for col in categorical_cols:
    if col not in df.columns:
        df[col] = "Unknown"
    df[col] = df[col].fillna("Unknown")

if "skills" not in df.columns:
    df["skills"] = ""
df["skills"] = df["skills"].fillna("").apply(
    lambda x: [item.strip() for item in str(x).split(",") if item.strip()]
)

for col in numeric_cols:
    if col not in df.columns:
        df[col] = 0
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# Encode categorical values
le_passion = LabelEncoder()
le_hobby = LabelEncoder()
le_edu = LabelEncoder()
le_career = LabelEncoder()

df["passion_enc"] = le_passion.fit_transform(df["passion"])
df["hobby_enc"] = le_hobby.fit_transform(df["hobby"])
df["edu_enc"] = le_edu.fit_transform(df["education_level"])
df["career_enc"] = le_career.fit_transform(df["career_title"])

all_skills = sorted({skill for skills in df["skills"] for skill in skills})
skill_matrix = pd.DataFrame(
    [[1 if skill in skills else 0 for skill in all_skills] for skills in df["skills"]],
    columns=all_skills,
)

interaction_features = pd.DataFrame({
    "passion_hobby_interaction": df["passion_enc"] * df["hobby_enc"],
    "age_edu_ratio": df["age"] / (df["edu_enc"] + 1),
    "openness_extraversion": df["openness"] * df["extraversion"] / 10,
    "conscientious_emotional": df["conscientiousness"] * (11 - df["neuroticism"]) / 10,
    "work_values_avg": (df["salary_importance"] + df["work_life_balance"]) / 2,
    "skill_count": df["skills"].apply(len),
})

feature_cols = [
    "passion_enc",
    "hobby_enc",
    "edu_enc",
    "age",
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
    "salary_importance",
    "work_life_balance",
    "passion_hobby_interaction",
    "age_edu_ratio",
    "openness_extraversion",
    "conscientious_emotional",
    "work_values_avg",
    "skill_count",
] + all_skills

X = pd.concat(
    [
        df[[
            "passion_enc",
            "hobby_enc",
            "edu_enc",
            "age",
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
            "salary_importance",
            "work_life_balance",
        ]],
        interaction_features,
        skill_matrix,
    ],
    axis=1,
)
y = df["career_enc"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y if len(set(y)) > 1 else None,
)

# Train model
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

model_candidates = [
    (
        "RandomForest",
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        {
            "model__n_estimators": [200, 300, 400, 500],
            "model__max_depth": [None, 10, 20, 30, 40],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 0.5, 0.7],
        },
    ),
    (
        "ExtraTrees",
        ExtraTreesClassifier(random_state=42, class_weight="balanced"),
        {
            "model__n_estimators": [300, 500, 700],
            "model__max_depth": [None, 10, 20, 30],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 0.5, 0.7],
        },
    ),
    (
        "GradientBoosting",
        GradientBoostingClassifier(random_state=42),
        {
            "model__n_estimators": [200, 300, 400],
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.7, 0.9, 1.0],
        },
    ),
    (
        "HistGradientBoosting",
        HistGradientBoostingClassifier(random_state=42),
        {
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__max_depth": [None, 6, 10],
            "model__max_iter": [200, 300],
            "model__l2_regularization": [0.0, 0.1, 0.5],
        },
    ),
]

best_search = None
best_score = -np.inf
best_name = None

for name, model, params in model_candidates:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model),
    ])

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=params,
        n_iter=12,
        scoring="f1_macro",
        cv=cv,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    if search.best_score_ > best_score:
        best_score = search.best_score_
        best_search = search
        best_name = name

best_model = best_search.best_estimator_.named_steps["model"]
scaler = best_search.best_estimator_.named_steps["scaler"]

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

best_model.fit(X_train_scaled, y_train)
test_preds = best_model.predict(X_test_scaled)

os.makedirs(MODELS_DIR, exist_ok=True)
model_path = os.path.join(MODELS_DIR, "career_rf_model.pkl")
scaler_path = os.path.join(MODELS_DIR, "feature_scaler.pkl")
encoders_path = os.path.join(MODELS_DIR, "sklearn_encoders.pkl")

encoders = {
    "le_passion": le_passion,
    "le_hobby": le_hobby,
    "le_edu": le_edu,
    "le_career": le_career,
    "all_skills": all_skills,
    "feature_cols": feature_cols,
}

# Save model, scaler, and encoders
joblib.dump(best_model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(encoders, encoders_path)

print(f"✅ Best model: {best_name}")
print(f"✅ Best CV macro F1: {best_score:.3f}")
print("✅ Test classification report:")
print(classification_report(y_test, test_preds))
print(f"✅ Best params: {best_search.best_params_}")
print(f"✅ Model trained and saved as '{model_path}'")
print(f"✅ Scaler saved as '{scaler_path}'")
print(f"✅ Encoders saved as '{encoders_path}'")
