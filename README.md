# Career Path Predictor - AI-Powered Career Guidance Platform

> "For those who come after to help them achieve their unseen dreams"

Career Path Predictor is a full-stack Flask app that uses a scikit-learn model to recommend career paths based on a user profile. It collects data from a multi-step assessment (passions, hobbies, education, skills, personality, work values, and aptitude), runs the profile through an ML pipeline, and renders the top 5 careers with confidence scores, salary/growth data, job links, and a generated learning roadmap. A downloadable PDF report is also available.

This README is intentionally exhaustive: it documents every major file, dataset, endpoint, UI flow, and ML step, plus includes real code snippets from the project.

---

## Quick Start

Prerequisites
- Python 3.8+
- pip

Run locally
```bash
pip install -r requirements.txt
python run.py
```

Open the app
```
http://localhost:5000
```

---

## What The App Does (End-to-End Flow)

1) Landing page
- Route: `/`
- Template: `templates/index.html`
- Purpose: introduces the product, shows features, and drives the user to the quiz.

2) Career assessment quiz
- Route: `/quiz`
- Template: `templates/quiz.html`
- Purpose: collects multi-step form inputs (passions, hobbies, education with specialization, skills, personality sliders, aptitude test, work values).

3) Prediction pipeline
- Route: `/predict` (POST)
- Logic: `app.py` (`predict()` -> `predict_career_with_confidence()`)
- Purpose: build a feature vector, scale it, compute probabilities, take top 5 careers, and augment results.

4) Results page
- Template: `templates/results.html`
- Purpose: show top 5 careers with confidence bars, salary in INR, skills, growth outlook, and job listings for the top result.

5) Career detail page
- Route: `/career/<career_name>`
- Template: `templates/career_details.html`
- Purpose: deep-dive into a single career and auto-generate a step-by-step learning path.

6) PDF report
- Route: `/download-pdf` (POST)
- Logic: `app.py` (`download_pdf()`)
- Purpose: exports the prediction summary to a PDF file and downloads it.

---

## Project Structure (Actual Files)

```
.
├── app.py
├── run.py
├── model.py
├── build_training_dataset.py
├── requirements.txt
├── datasets/
│   ├── career.csv
│   ├── passion.csv
│   ├── hobbies.csv
│   ├── skills.csv
│   ├── personality.csv
│   ├── work_values.csv
│   ├── training_data_clean.csv
│   ├── training_data_comprehensive.csv
│   └── ...
├── models/
│   ├── career_rf_model.pkl
│   ├── feature_scaler.pkl
│   └── sklearn_encoders.pkl
├── static/
│   ├── theme.css
│   └── theme.js
└── templates/
   ├── index.html
   ├── quiz.html
   ├── results.html
   ├── career_details.html
   ├── error.html
   ├── privacy_policy.html
   └── terms_of_service.html
```

---

## Tech Stack

- Backend: Flask 3.0
- ML: scikit-learn 1.3, NumPy 1.24, pandas 2.1
- Frontend: HTML, CSS, JavaScript, Bootstrap 5
- Data: CSV datasets (skills, passions, careers, training data)
- PDF: fpdf
- Web scraping: requests + BeautifulSoup (job links)

Dependencies in `requirements.txt`:
```
Flask==3.0.0
pandas==2.1.0
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.2
fpdf==1.7.2
Werkzeug==3.0.1
requests==2.31.0
beautifulsoup4==4.12.2
lxml==4.9.3
```

---

## Backend Architecture (Flask)

### App initialization
`app.py` bootstraps Flask, loads model artifacts, and loads CSV datasets.

Key configuration:
- `MODEL_PATH` -> `models/career_rf_model.pkl`
- `SCALER_PATH` -> `models/feature_scaler.pkl`
- `ENCODERS_PATH` -> `models/sklearn_encoders.pkl`
- `YOUTUBE_API_KEY` (optional)

Snippet (model loading and fallback):
```python
def load_model_artifacts():
   try:
      loaded_model = joblib.load(MODEL_PATH)
      loaded_scaler = joblib.load(SCALER_PATH)
      loaded_encoders = joblib.load(ENCODERS_PATH)
      return loaded_model, loaded_scaler, loaded_encoders
   except Exception:
      subprocess.run([sys.executable, TRAINING_SCRIPT_PATH], cwd=BASE_DIR, check=True)
      loaded_model = joblib.load(MODEL_PATH)
      loaded_scaler = joblib.load(SCALER_PATH)
      loaded_encoders = joblib.load(ENCODERS_PATH)
      return loaded_model, loaded_scaler, loaded_encoders
```

Note: `TRAINING_SCRIPT_PATH` points to `model_enhanced.py`. That file is referenced in code but is not included in this repository. If you want rebuilds to work, either add `model_enhanced.py` or update the path to `model.py` (which is present).

---

### Routes & pages

- `/` -> `index()` -> landing page
- `/quiz` -> `quiz()` -> assessment form
- `/predict` (POST) -> `predict()` -> scoring pipeline
- `/career/<career_name>` -> `career_detail()` -> deep-dive view
- `/download-pdf` (POST) -> `download_pdf()` -> export
- `/privacy-policy` -> `privacy_policy()`
- `/terms-of-service` -> `terms_of_service()`

Error handlers:
- 404 -> `error.html`
- 500 -> `error.html`

---

## Prediction Pipeline (Detailed)

### 1) Input capture
`/quiz` collects:
- passion (single select)
- hobbies (multi-select, at least 1)
- education + specialization
- age
- skills (multi-select, at least 3)
- Big Five personality traits (1-10 sliders)
- aptitude test (10 questions)
- work values (salary importance, work-life balance)

### 2) Education normalization
High school and bachelor specialization are captured for display, but the model expects only broad categories.

Snippet:
```python
def process_education_input(education_level, hs_grade=None, hs_combination=None,
                     bachelor_type=None, bachelor_spec=None):
   if education_level == "High School" and hs_combination:
      display = f"High School - {hs_grade} ({hs_combination})"
      return "High School", display
   elif education_level == "Bachelor's" and bachelor_spec:
      return "Bachelor's", bachelor_spec
   return education_level, None
```

### 3) Feature building
`predict_career_with_confidence()` encodes categorical values, adds numeric features, generates interaction features, and appends a skill vector.

Snippet (feature build & top-5 selection):
```python
features = [
   passion_enc, hobby_enc, edu_enc, input_data['age'],
   input_data.get('openness', 5), input_data.get('conscientiousness', 5),
   input_data.get('extraversion', 5), input_data.get('agreeableness', 5),
   input_data.get('neuroticism', 5), input_data.get('salary_importance', 5),
   input_data.get('work_life_balance', 5),
]

features.extend([
   passion_enc * hobby_enc,
   input_data['age'] / (edu_enc + 1),
   features[4] * features[6] / 10,
   features[5] * (11 - features[8]) / 10,
   (features[9] + features[10]) / 2,
])

user_skills = set(input_data.get('skills', []))
skill_features = [1 if skill in user_skills else 0 for skill in encoders['all_skills']]
features.append(len(user_skills))
features.extend(skill_features)

features_scaled = scaler.transform([features])
probabilities = model.predict_proba(features_scaled)[0]

top_5_indices = np.argsort(probabilities)[-5:][::-1]
top_5_careers = encoders['le_career'].inverse_transform(top_5_indices)
top_5_probs = probabilities[top_5_indices]
```

### 4) Result enhancement
Each top career is enriched with data from `datasets/career.csv`:
- median salary (converted to INR)
- education level
- growth outlook
- skills required

Salary conversion uses a fixed USD->INR rate:
```python
USD_TO_INR = 83.0
```

### 5) Jobs and videos
- `fetch_linkedin_jobs()` scrapes Google results for links; falls back to curated entries.
- `fetch_youtube_videos()` uses YouTube Data API (requires `YOUTUBE_API_KEY`).

---

## ML Training Pipeline

There are two data preparation scripts in this repo:

### build_training_dataset.py
Purpose: clean and standardize `training_data_comprehensive.csv` and emit `training_data_clean.csv`.

Highlights:
- Normalizes education labels (e.g., "masters" -> "Master's")
- Normalizes skills and removes duplicates
- Uses reference lists from `passion.csv`, `hobbies.csv`, `skills.csv`, and `career.csv`
- Drops incomplete rows and duplicates

Snippet (skill normalization):
```python
def normalize_skills(value: str) -> str:
   if not isinstance(value, str):
      return value
   parts = [item.strip() for item in value.split(",") if item.strip()]
   return ", ".join(sorted(set(parts)))
```

Run it:
```bash
python build_training_dataset.py
```

### model.py
Purpose: train and persist the best-performing classifier.

What it does:
- Loads `training_data_clean.csv` (or `training_data_comprehensive.csv` if clean is missing).
- Encodes categorical fields: passion, hobby, education.
- Builds a skill binary matrix for multi-skill input.
- Adds interaction features (passion*hobby, age/edu ratio, etc.).
- Trains multiple candidates (RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting).
- Runs randomized search with F1 macro scoring and 3-fold stratified CV.
- Saves the best model + scaler + encoders into `models/`.

Snippet (interaction features):
```python
interaction_features = pd.DataFrame({
   "passion_hobby_interaction": df["passion_enc"] * df["hobby_enc"],
   "age_edu_ratio": df["age"] / (df["edu_enc"] + 1),
   "openness_extraversion": df["openness"] * df["extraversion"] / 10,
   "conscientious_emotional": df["conscientiousness"] * (11 - df["neuroticism"]) / 10,
   "work_values_avg": (df["salary_importance"] + df["work_life_balance"]) / 2,
   "skill_count": df["skills"].apply(len),
})
```

Train the model:
```bash
python model.py
```

Artifacts saved:
- `models/career_rf_model.pkl`
- `models/feature_scaler.pkl`
- `models/sklearn_encoders.pkl`

---

## Frontend Structure

### templates/index.html
Landing page with:
- Hero section and CTA
- Features grid
- Mission quote and About section
- Footer links to privacy and terms

### templates/quiz.html
Multi-step form with client-side validation:
- Step 1: passion, hobbies, education, age
- Step 2: skills (min 3)
- Step 3: Big Five personality sliders
- Step 4: aptitude test (10 questions)
- Step 5: work values sliders

The aptitude section calculates a 0-10 score and stores it in a hidden field:
```javascript
let score = 0;
for (let i = 1; i <= 10; i++) {
   const sel = document.querySelector(`input[name=q${i}]:checked`);
   if (sel && sel.value) score += parseInt(sel.value);
}
document.getElementById('aptitude_score').value = score;
```

### templates/results.html
- Renders top 5 predictions with confidence bars
- Shows salary (INR), education, growth outlook, skills
- Shows job listings for the top career
- Button to download PDF and to view a detailed learning path

### templates/career_details.html
Displays:
- Career summary from `career.csv`
- A generated learning path based on skill gaps

Learning path generation is purely heuristic (not ML):
```python
skill_gaps = [skill for skill in required_skills if skill not in user_skills]
path["steps"].append({
   "title": "Gain Practical Experience",
   "resources": ["Internships", "Freelance projects", "Open source contributions"],
})
```

### static/theme.css + static/theme.js
Provides the dual theme system:
- Default theme: seabreeze
- Alternate theme: dark
- Smooth transitions and animated toggle

`theme.js` stores preferences in localStorage under `careerpath-theme`.

---

## Data Files (datasets/)

These CSV files drive both training and UI selections:

- `career.csv` - master career catalog, salary, growth, skills, education, industry.
- `passion.csv` - list of passions for the quiz.
- `hobbies.csv` - list of hobbies for the quiz.
- `skills.csv` - list of skill tags for the quiz.
- `personality.csv` - optional extra traits merged during cleaning.
- `work_values.csv` - reference list of work value attributes.
- `training_data_comprehensive.csv` - raw ML training dataset.
- `training_data_clean.csv` - cleaned and normalized dataset generated by script.

---

## PDF Report Generation

`/download-pdf` uses fpdf to create a compact summary of the top 5 results.

Snippet:
```python
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", "B", 24)
pdf.cell(0, 20, "Career Path Report", ln=True, align="C")
```

---

## Environment Variables

- `YOUTUBE_API_KEY` (optional)
  - When set, `/career/<career_name>` will embed top YouTube videos.
  - If missing, no videos are displayed.

---

## Known Design Choices & Limits

- Education granularity in UI is richer than model input. The model only uses two categories: High School and Bachelor's.
- Only the first selected hobby is used for model input; all hobbies are retained for display and session.
- Salary in results is converted from USD to INR using a static rate (83.0) without live forex.
- Job links are scraped from Google search; if blocked, curated sample links are shown.
- Aptitude score is currently displayed but not used as a model feature.

---

## How To Retrain Safely

1) Clean data (optional but recommended):
```bash
python build_training_dataset.py
```

2) Train model and export artifacts:
```bash
python model.py
```

3) Start the app:
```bash
python run.py
```

---

## Screenshots

Screenshots are stored in `screenshots/` and are referenced by `README.md` or can be embedded elsewhere.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

Portfolio: https://ashraf-portfolio49.vercel.app/
