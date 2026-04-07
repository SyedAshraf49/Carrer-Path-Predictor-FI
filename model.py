# model.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# Load CSVs
career = pd.read_csv("career.csv")
freelance = pd.read_csv("freelance.csv")
hobbies = pd.read_csv("hobbies.csv")
large_age = pd.read_csv("large_age.csv")
major_career = pd.read_csv("major_career.csv")
minor_career = pd.read_csv("minor_career.csv")
passion = pd.read_csv("passion.csv")

# Prepare synthetic training dataset
df = pd.DataFrame({
    "passion": passion["passion"],
    "hobby": hobbies["hobby"],
    "education_level": career["education_level"][:30].values,
    "age": large_age["Age"][:30].values,
    "skills": career["skills_required"][:30].apply(lambda x: x.split(", ")),
    "career_title": career["career_title"]
})

# Encode categorical values
le_passion = LabelEncoder()
le_hobby = LabelEncoder()
le_edu = LabelEncoder()
le_career = LabelEncoder()
mlb_skills = MultiLabelBinarizer()

df["passion_enc"] = le_passion.fit_transform(df["passion"])
df["hobby_enc"] = le_hobby.fit_transform(df["hobby"])
df["edu_enc"] = le_edu.fit_transform(df["education_level"])
df["career_enc"] = le_career.fit_transform(df["career_title"])
skills_encoded = mlb_skills.fit_transform(df["skills"])

# Combine into final input feature set
X = pd.concat([df[["passion_enc", "hobby_enc", "edu_enc", "age"]], pd.DataFrame(skills_encoded)], axis=1)
y = df["career_enc"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model and encoders
joblib.dump(clf, "career_rf_model.pkl")
joblib.dump((le_passion, le_hobby, le_edu, le_career, mlb_skills), "sklearn_encoders.pkl")

print("✅ Model trained and saved as 'career_rf_model.pkl'")
print("✅ Encoders saved as 'sklearn_encoders.pkl'")
