# app.py - Updated with multiple hobbies and high school specializations
"""
Complete Flask app with multiple hobbies, high school specializations, and INR currency
"""

from flask import Flask, render_template, request, send_file, session, redirect, url_for
import joblib
import numpy as np
import pandas as pd
from fpdf import FPDF
import os
import subprocess
import sys
from datetime import datetime
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
app.secret_key = "career-predictor-secret-key-2024"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "career_rf_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "feature_scaler.pkl")
ENCODERS_PATH = os.path.join(MODELS_DIR, "sklearn_encoders.pkl")
TRAINING_SCRIPT_PATH = os.path.join(BASE_DIR, "model_enhanced.py")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/search"

# USD to INR conversion rate (approximate)
USD_TO_INR = 83.0

# ==================== EDUCATION MAPPINGS ====================

# High School Grade Options
HIGH_SCHOOL_GRADES = ["11th Grade", "12th Grade"]

# High School Subject Combinations
HIGH_SCHOOL_COMBINATIONS = [
    "Comp.Sci/Economics/Commerce/Accountancy",
    "Physics/Chemistry/Botany/Zoology",
    "Physics/Chemistry/Comp.Science/Mathematics",
    "Economics/Commerce/Accountancy/Busi.Maths",
    "History/Economics/Commerce/Accountancy"
]

# Bachelor's Options
BACHELOR_ARTS_OPTIONS = [
    "B.Com.",
    "B.Com. C.A.",
    "B.Com. E-Commerce",
    "B.Com. Corporate Secretaryship",
    "B.Com. Corporate Secretaryship with C.A.",
    "B.Com. Retail Marketing",
    "B.Com. Information Technology",
    "B.Com. Banking & Insurance",
    "B.Com. Co-operation"
]

BACHELOR_SCIENCE_OPTIONS = [
    "B.Sc. Mathematics",
    "B.Sc. Statistics",
    "B.Sc. Physics",
    "B.Sc. Physics with Nano-Technology",
    "B.Sc. Physics with C.A.",
    "B.Sc. Plant Biology and Plant Biotechnology / B.Sc. Botany",
    "B.Sc. Chemistry",
    "B.Sc. Biochemistry",
    "B.Sc. Pharmaceutical Chemistry",
    "B.Sc. Polymer Technology",
    "B.C.A.",
    "B.Sc. Computer Science",
    "B.Sc. Information Technology",
    "B.Sc. Software Systems",
    "B.Sc. Computer Technology",
    "B.Sc. Multimedia & Web Technology",
    "B.Sc. Clinical Laboratory Technology",
    "B.Sc. Electronics & Communication System",
    "B.Sc. Biotechnology",
    "B.Sc. Interior Design with Computer Applications",
    "B.Sc. Microbiology with Nanotechnology",
    "B.Sc. Geography",
    "B.Sc. Advanced Zoology and Biotech with Sericulture",
    "B.Sc. Advanced Zoology and Biotechnology",
    "B.Sc. Zoology (Wildlife Biology)",
    "B.Sc. Costume Design and Fashion",
    "B.Sc. Fashion Apparel Management",
    "B.Sc. Nutrition & Dietetics",
    "B.Sc. Food Science & Nutrition / B.Sc. Food Science & Nutrition with CA",
    "B.Sc. Visual Communication & Electronic Media"
]

# ==================== LOAD MODELS AND DATA ====================
print("🔄 Loading models and data...")

def load_model_artifacts():
    """Load persisted model artifacts, rebuilding them if the pickle is incompatible."""

    try:
        loaded_model = joblib.load(MODEL_PATH)
        loaded_scaler = joblib.load(SCALER_PATH)
        loaded_encoders = joblib.load(ENCODERS_PATH)
        print("✅ Models loaded successfully")
        return loaded_model, loaded_scaler, loaded_encoders
    except FileNotFoundError as e:
        print(f"❌ Error loading models: {e}")
        print("⚠️  Missing model artifacts. Regenerating them now...")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("⚠️  Existing model artifacts are incompatible with the current scikit-learn version.")
        print("⚠️  Rebuilding them now with the local environment...")

    try:
        print(f"⚠️  Attempting to rebuild models using {TRAINING_SCRIPT_PATH}...")
        result = subprocess.run([sys.executable, TRAINING_SCRIPT_PATH], cwd=BASE_DIR, check=False, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"⚠️  Model rebuild script output:")
            print(result.stdout)
            print(result.stderr)
            print(f"⚠️  Model rebuild failed with exit code {result.returncode}")
        else:
            loaded_model = joblib.load(MODEL_PATH)
            loaded_scaler = joblib.load(SCALER_PATH)
            loaded_encoders = joblib.load(ENCODERS_PATH)
            print("✅ Rebuilt and loaded models successfully")
            return loaded_model, loaded_scaler, loaded_encoders
    except subprocess.TimeoutExpired:
        print(f"❌ Model rebuild timed out after 300 seconds")
    except Exception as rebuild_error:
        print(f"❌ Failed to rebuild models: {rebuild_error}")
        import traceback
        traceback.print_exc()
    
    print("⚠️  Models not available. Using fallback prediction mode.")
    return None, None, None


model, scaler, encoders = load_model_artifacts()

try:
    career_df = pd.read_csv("datasets/career.csv")
    career_details = career_df.set_index("career_title").to_dict('index')
    
    passion_df = pd.read_csv("datasets/passion.csv")
    hobbies_df = pd.read_csv("datasets/hobbies.csv")
    skills_df = pd.read_csv("datasets/skills.csv")
    
    passions = passion_df["passion"].tolist()
    hobbies = hobbies_df["hobby"].tolist()
    available_skills = skills_df["skill_name"].tolist()
    
    print("✅ Data loaded successfully")
except FileNotFoundError as e:
    print(f"❌ Error loading data: {e}")
    career_details = {}
    passions = []
    hobbies = []
    available_skills = []

# Updated education levels (removed Vocational, Associate's, Master's, Doctorate)
education_levels = ["High School", "Bachelor's"]

# ==================== HELPER FUNCTIONS ====================

def convert_salary_to_inr(salary_str):
    """Convert USD salary to INR format"""
    try:
        salary_usd = int(salary_str.replace('$', '').replace(',', ''))
        salary_inr = salary_usd * USD_TO_INR
        
        if salary_inr >= 100000:
            lakhs = salary_inr / 100000
            return f"₹{lakhs:.2f} Lakhs"
        else:
            return f"₹{salary_inr:,.0f}"
    except:
        return "N/A"

def process_education_input(education_level, hs_grade=None, hs_combination=None, 
                           bachelor_type=None, bachelor_spec=None):
    """
    Process education input to handle High School and Bachelor's specializations
    Returns the education string to use for model prediction and display
    """
    if education_level == "High School" and hs_combination:
        # For display, include grade and combination
        display = f"High School - {hs_grade} ({hs_combination})"
        # For model, just use "High School"
        return "High School", display
    elif education_level == "Bachelor's" and bachelor_spec:
        # For display, use full specialization
        return "Bachelor's", bachelor_spec
    else:
        return education_level, None

# ==================== ROUTES ====================

@app.route("/")
def index():
    """Home page - Landing page"""
    return render_template("index.html")

@app.route("/quiz")
def quiz():
    """Assessment quiz page"""
    return render_template("quiz.html", 
                         passions=passions,
                         hobbies=hobbies,
                         educations=education_levels,
                         skills=available_skills,
                         hs_grades=HIGH_SCHOOL_GRADES,
                         hs_combinations=HIGH_SCHOOL_COMBINATIONS,
                         bachelor_arts=BACHELOR_ARTS_OPTIONS,
                         bachelor_science=BACHELOR_SCIENCE_OPTIONS)

@app.route("/privacy-policy")
def privacy_policy():
    """Privacy Policy page"""
    return render_template("privacy_policy.html")

@app.route("/terms-of-service")
def terms_of_service():
    """Terms of Service page"""
    return render_template("terms_of_service.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Process form and predict careers"""
    
    if model is None or scaler is None or encoders is None:
        print("⚠️  Models not available - using fallback predictions")
        # Continue with fallback mode instead of erroring out
    
    try:
        # Extract form data
        passion = request.form.get("passion")
        
        # Get multiple hobbies
        selected_hobbies = request.form.getlist("hobbies")
        if not selected_hobbies:
            return render_template("error.html", 
                                 error="Please select at least one hobby")
        # Use first hobby for model (or we could combine them)
        hobby = selected_hobbies[0]
        
        education_level = request.form.get("education_level")
        
        # High School fields
        hs_grade = request.form.get("hs_grade", "")
        hs_combination = request.form.get("hs_combination", "")
        
        # Bachelor's fields
        bachelor_type = request.form.get("bachelor_type", "")
        bachelor_spec = request.form.get("bachelor_specialization", "")
        
        age = int(request.form.get("age"))
        skills = request.form.getlist("skills")
        
        # Process education
        education_for_model, full_education = process_education_input(
            education_level, hs_grade, hs_combination, bachelor_type, bachelor_spec
        )
        
        # Get personality traits
        openness = int(request.form.get("openness", 5))
        conscientiousness = int(request.form.get("conscientiousness", 5))
        extraversion = int(request.form.get("extraversion", 5))
        agreeableness = int(request.form.get("agreeableness", 5))
        neuroticism = int(request.form.get("neuroticism", 5))
        
        # Get work values
        salary_importance = int(request.form.get("salary_importance", 5))
        work_life_balance = int(request.form.get("work_life_balance", 5))
        
        # Get aptitude score if provided
        aptitude_score = None
        try:
            aptitude_score = int(request.form.get("aptitude_score")) if request.form.get("aptitude_score") else None
        except (ValueError, TypeError):
            aptitude_score = None
        
        # Build input data dictionary
        input_data = {
            "passion": passion,
            "hobby": hobby,  # Primary hobby for model
            "hobbies": selected_hobbies,  # Store all hobbies
            "education": education_for_model,
            "education_display": full_education or education_level,
            "age": age,
            "skills": skills,
            "openness": openness,
            "conscientiousness": conscientiousness,
            "extraversion": extraversion,
            "agreeableness": agreeableness,
            "neuroticism": neuroticism,
            "salary_importance": salary_importance,
            "work_life_balance": work_life_balance
        }
        
        # Get predictions
        predictions = predict_career_with_confidence(input_data)
        
        # Enhance predictions with career details (convert to INR)
        enhanced_predictions = []
        for career, confidence in predictions:
            career_info = career_details.get(career, {})
            
            salary_usd = career_info.get("median_salary_usd", "N/A")
            salary_inr = convert_salary_to_inr(salary_usd) if salary_usd != "N/A" else "N/A"
            
            enhanced_predictions.append({
                "career": career,
                "confidence": float(confidence),
                "confidence_percent": f"{confidence*100:.1f}%",
                "salary": salary_inr,
                "education": career_info.get("education_level", "N/A"),
                "growth": career_info.get("job_growth_outlook", "N/A"),
                "skills": career_info.get("skills_required", "N/A")
            })
        
        # Fetch job listings for top career
        jobs = []
        try:
            if enhanced_predictions:
                top_career = enhanced_predictions[0]['career']
                jobs = fetch_linkedin_jobs(top_career, limit=6)
        except Exception as e:
            print(f"Warning: Could not fetch jobs: {e}")
            jobs = []
        
        # Store in session
        session['last_prediction'] = enhanced_predictions[0]['career']
        session['user_profile'] = input_data
        session['all_predictions'] = enhanced_predictions
        session['aptitude_score'] = aptitude_score
        
        return render_template("results.html", predictions=enhanced_predictions, aptitude_score=aptitude_score, jobs=jobs)
        
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return render_template("error.html", 
                             error=f"An error occurred during prediction: {str(e)}")

@app.route("/career/<career_name>")
def career_detail(career_name):
    """Detailed career information page"""
    
    career_info = career_details.get(career_name, {})
    
    if not career_info:
        return render_template("error.html", 
                             error=f"Career '{career_name}' not found in database.")
    
    # Convert salary to INR
    salary_usd = career_info.get("median_salary_usd", "N/A")
    career_info_display = career_info.copy()
    career_info_display["median_salary_inr"] = convert_salary_to_inr(salary_usd)
    
    # Generate learning path
    user_profile = session.get('user_profile', {})
    learning_path = generate_learning_path(career_name, user_profile)

    youtube_videos = fetch_youtube_videos(career_name, limit=5)
    
    return render_template("career_details.html",
                         career=career_name,
                         info=career_info_display,
                         learning_path=learning_path,
                         youtube_videos=youtube_videos)

@app.route("/download-pdf", methods=["POST"])
def download_pdf():
    """Generate and download PDF report"""
    
    try:
        predictions = session.get('all_predictions', [])
        
        if not predictions:
            return render_template("error.html", 
                                 error="No predictions found. Please take the assessment first.")
        
        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 24)
        
        # Title
        pdf.cell(0, 20, "Career Path Report", ln=True, align="C")
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 10, f"Generated on {datetime.now().strftime('%B %d, %Y')}", ln=True, align="C")
        pdf.ln(10)
        
        # Add predictions
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Your Top 5 Career Matches", ln=True)
        pdf.ln(5)
        
        for i, pred in enumerate(predictions, 1):
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, f"{i}. {pred['career']} - {pred['confidence_percent']} Match", ln=True)
            
            pdf.set_font("Arial", "", 10)
            pdf.cell(0, 6, f"   Salary: {pred['salary']}", ln=True)
            pdf.cell(0, 6, f"   Education: {pred['education']}", ln=True)
            pdf.cell(0, 6, f"   Growth: {pred['growth']}", ln=True)
            pdf.ln(3)
        
        # Save PDF
        pdf_filename = f"career_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(pdf_filename)
        
        return send_file(pdf_filename, as_attachment=True, download_name="CareerPath_Report.pdf")
        
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        return render_template("error.html", 
                             error=f"Error generating PDF: {str(e)}")

@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", error="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", error="Internal server error"), 500

# ==================== PREDICTION HELPERS ====================

def predict_career_with_confidence(input_data):
    """Predict top 5 careers with confidence scores"""
    
    # If models not available, return fallback predictions
    if model is None or scaler is None or encoders is None:
        print("⚠️  Models not available - returning fallback predictions")
        return [
            ("Software Engineer", 0.85),
            ("Data Scientist", 0.78),
            ("Graphic Designer", 0.72),
            ("Teacher", 0.68),
            ("Marketing Manager", 0.65)
        ]
    
    try:
        # Encode categorical inputs
        passion_enc = encoders['le_passion'].transform([input_data['passion']])[0]
        hobby_enc = encoders['le_hobby'].transform([input_data['hobby']])[0]
        edu_enc = encoders['le_edu'].transform([input_data['education']])[0]
        
        # Build feature vector
        features = [
            passion_enc,
            hobby_enc,
            edu_enc,
            input_data['age'],
            input_data.get('openness', 5),
            input_data.get('conscientiousness', 5),
            input_data.get('extraversion', 5),
            input_data.get('agreeableness', 5),
            input_data.get('neuroticism', 5),
            input_data.get('salary_importance', 5),
            input_data.get('work_life_balance', 5),
        ]
        
        # Add interaction features
        features.extend([
            passion_enc * hobby_enc,
            input_data['age'] / (edu_enc + 1),
            features[4] * features[6] / 10,
            features[5] * (11 - features[8]) / 10,
            (features[9] + features[10]) / 2,
        ])
        
        # Add skill features
        user_skills = set(input_data.get('skills', []))
        skill_features = [1 if skill in user_skills else 0 for skill in encoders['all_skills']]
        features.append(len(user_skills))
        features.extend(skill_features)
        
        # Ensure correct feature length
        expected_length = len(encoders['feature_cols'])
        if len(features) < expected_length:
            features.extend([0] * (expected_length - len(features)))
        elif len(features) > expected_length:
            features = features[:expected_length]
        
        # Scale and predict
        features_scaled = scaler.transform([features])
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Get top 5
        top_5_indices = np.argsort(probabilities)[-5:][::-1]
        top_5_careers = encoders['le_career'].inverse_transform(top_5_indices)
        top_5_probs = probabilities[top_5_indices]
        
        return list(zip(top_5_careers, top_5_probs))
        
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return [
            ("Software Engineer", 0.85),
            ("Data Scientist", 0.78),
            ("Graphic Designer", 0.72),
            ("Teacher", 0.68),
            ("Marketing Manager", 0.65)
        ]

def fetch_linkedin_jobs(title, location="", limit=5):
    """Fetch job listings - returns curated sample jobs by career type.
    
    Returns a list of {title, company, link} for the given job title.
    """
    try:
        # Try Google Jobs search first
        search_url = f"https://www.google.com/search?q={requests.utils.quote(title + ' jobs')}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        resp = requests.get(search_url, headers=headers, timeout=8)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            results = []
            
            for item in soup.find_all('div', class_='g'):
                title_elem = item.find('h3')
                link_elem = item.find('a', href=True)
                
                if title_elem and link_elem:
                    job_title = title_elem.get_text(strip=True)
                    link = link_elem.get('href', '')
                    
                    if 'job' in job_title.lower() or 'linkedin' in link.lower():
                        results.append({
                            'title': job_title[:80],
                            'company': 'Various Companies',
                            'link': link
                        })
                
                if len(results) >= limit:
                    break
            
            if results:
                return results
    except Exception as e:
        print(f"Job search error: {e}")
    
    # Fallback: Return curated sample jobs based on title
    sample_jobs = {
        'Software Engineer': [
            {'title': 'Senior Software Engineer - Python', 'company': 'Tech Corp', 'link': 'https://www.linkedin.com/jobs/search/?keywords=Software%20Engineer'},
            {'title': 'Full Stack Developer', 'company': 'StartUp Inc', 'link': 'https://www.linkedin.com/jobs/search/?keywords=Full%20Stack%20Developer'},
            {'title': 'Backend Engineer - Java', 'company': 'Cloud Systems', 'link': 'https://www.linkedin.com/jobs/search/?keywords=Backend%20Engineer'},
            {'title': 'Frontend React Developer', 'company': 'WebDev Labs', 'link': 'https://www.linkedin.com/jobs/search/?keywords=React%20Developer'},
            {'title': 'DevOps Engineer', 'company': 'Infrastructure Co', 'link': 'https://www.linkedin.com/jobs/search/?keywords=DevOps%20Engineer'},
        ],
        'Data Scientist': [
            {'title': 'Senior Data Scientist', 'company': 'Analytics Pro', 'link': 'https://www.linkedin.com/jobs/search/?keywords=Data%20Scientist'},
            {'title': 'Machine Learning Engineer', 'company': 'AI Solutions', 'link': 'https://www.linkedin.com/jobs/search/?keywords=Machine%20Learning%20Engineer'},
            {'title': 'Data Analyst', 'company': 'Business Intel', 'link': 'https://www.linkedin.com/jobs/search/?keywords=Data%20Analyst'},
            {'title': 'Research Scientist', 'company': 'Tech Labs', 'link': 'https://www.linkedin.com/jobs/search/?keywords=Research%20Scientist'},
            {'title': 'Analytics Manager', 'company': 'Insights Corp', 'link': 'https://www.linkedin.com/jobs/search/?keywords=Analytics%20Manager'},
        ],
        'Product Manager': [
            {'title': 'Senior Product Manager', 'company': 'Product Co', 'link': 'https://www.linkedin.com/jobs/search/?keywords=Product%20Manager'},
            {'title': 'Associate Product Manager', 'company': 'Tech Solutions', 'link': 'https://www.linkedin.com/jobs/search/?keywords=Associate%20Product%20Manager'},
            {'title': 'Product Strategy Lead', 'company': 'Innovation Labs', 'link': 'https://www.linkedin.com/jobs/search/?keywords=Product%20Strategy'},
            {'title': 'Product Operations Manager', 'company': 'Operations Plus', 'link': 'https://www.linkedin.com/jobs/search/?keywords=Product%20Operations'},
            {'title': 'Technical Product Manager', 'company': 'Tech Innovations', 'link': 'https://www.linkedin.com/jobs/search/?keywords=Technical%20Product%20Manager'},
        ],
    }
    
    # Try to find matching jobs for the title
    for key in sample_jobs:
        if key.lower() in title.lower() or title.lower() in key.lower():
            return sample_jobs[key][:limit]
    
    # Default: return generic tech jobs
    return sample_jobs['Software Engineer'][:limit]


def fetch_youtube_videos(career_title, limit=5):
    """Fetch top YouTube videos for a career title using the YouTube Data API."""
    if not YOUTUBE_API_KEY:
        return []

    query = f"{career_title} career roadmap skills"
    params = {
        "key": YOUTUBE_API_KEY,
        "q": query,
        "part": "snippet",
        "type": "video",
        "maxResults": limit,
        "safeSearch": "strict",
    }

    try:
        resp = requests.get(YOUTUBE_API_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])
        videos = []
        for item in items:
            video_id = item.get("id", {}).get("videoId")
            snippet = item.get("snippet", {})
            if not video_id:
                continue
            videos.append({
                "title": snippet.get("title", "Untitled"),
                "channel": snippet.get("channelTitle", ""),
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "thumbnail": snippet.get("thumbnails", {}).get("medium", {}).get("url", ""),
            })
        return videos
    except Exception as exc:
        print(f"Warning: Could not fetch YouTube videos: {exc}")
        return []

def generate_learning_path(career_name, user_profile):
    """Generate personalized learning roadmap"""
    
    career_info = career_details.get(career_name, {})
    required_skills = career_info.get("skills_required", "").split(", ") if career_info.get("skills_required") else []
    required_education = career_info.get("education_level", "Bachelor's")
    
    user_skills = set(user_profile.get("skills", []))
    user_education = user_profile.get("education", "High School")
    
    # Skill gap analysis
    skill_gaps = [skill for skill in required_skills if skill not in user_skills]
    
    # Education gap
    education_levels_list = ["High School", "Bachelor's"]
    user_edu_level = education_levels_list.index(user_education) if user_education in education_levels_list else 0
    required_edu_level = education_levels_list.index(required_education) if required_education in education_levels_list else 0
    needs_more_education = user_edu_level < required_edu_level
    
    # Build learning path
    path = {
        "career": career_name,
        "current_match": f"{len(user_skills & set(required_skills))}/{len(required_skills)} skills",
        "timeline": "6-24 months" if len(skill_gaps) <= 3 else "24-48 months",
        "steps": []
    }
    
    step_num = 1
    
    # Education steps
    if needs_more_education:
        path["steps"].append({
            "step": step_num,
            "title": f"Complete {required_education}",
            "description": f"Upgrade from {user_education} to {required_education}",
            "duration": "3-4 years",
            "resources": ["University programs", "Online degrees", "Community colleges"]
        })
        step_num += 1
    
    # Skill acquisition steps (top 5)
    for skill in skill_gaps[:5]:
        path["steps"].append({
            "step": step_num,
            "title": f"Learn {skill}",
            "description": f"Acquire proficiency in {skill}",
            "duration": "2-6 months",
            "resources": ["Coursera", "Udemy", "YouTube tutorials", "Practice projects"]
        })
        step_num += 1
    
    # Experience step
    path["steps"].append({
        "step": step_num,
        "title": "Gain Practical Experience",
        "description": "Build portfolio and get real-world experience",
        "duration": "3-12 months",
        "resources": ["Internships", "Freelance projects", "Open source contributions", "Personal projects"]
    })
    step_num += 1
    
    # Certification step
    path["steps"].append({
        "step": step_num,
        "title": "Earn Certifications",
        "description": "Get industry-recognized credentials",
        "duration": "1-6 months",
        "resources": ["Professional certifications", "Industry certificates", "Online credentials"]
    })
    
    return path

# ==================== RUN APP ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting Career Path Predictor")
    print("="*60)
    print("📍 Open your browser to: http://localhost:5000")
    print("🛑 Press CTRL+C to stop the server")
    print("="*60 + "\n")
    
    port = int(os.getenv("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)