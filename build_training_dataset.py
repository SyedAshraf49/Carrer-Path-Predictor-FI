import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets")

INPUT_MAIN = os.path.join(DATA_DIR, "training_data_comprehensive.csv")
OPTIONAL_PERSONALITY = os.path.join(DATA_DIR, "personality.csv")
OUTPUT_CLEAN = os.path.join(DATA_DIR, "training_data_clean.csv")
PASSIONS_PATH = os.path.join(DATA_DIR, "passion.csv")
HOBBIES_PATH = os.path.join(DATA_DIR, "hobbies.csv")
SKILLS_PATH = os.path.join(DATA_DIR, "skills.csv")
CAREERS_PATH = os.path.join(DATA_DIR, "career.csv")

EDU_MAP = {
    "bachelors": "Bachelor's",
    "bachelor": "Bachelor's",
    "masters": "Master's",
    "master": "Master's",
    "phd": "Doctorate",
    "doctorate": "Doctorate",
    "high school": "High School",
    "associate": "Associate's",
    "associates": "Associate's",
    "vocational": "Vocational",
}

NUMERIC_COLUMNS = [
    "age",
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
    "salary_importance",
    "work_life_balance",
    "risk_tolerance",
    "creativity_score",
    "analytical_thinking",
    "people_orientation",
    "detail_oriented",
    "stress_tolerance",
]


def normalize_education(value: str) -> str:
    if not isinstance(value, str):
        return value
    key = value.strip().lower()
    return EDU_MAP.get(key, value.strip())


def normalize_skills(value: str) -> str:
    if not isinstance(value, str):
        return value
    parts = [item.strip() for item in value.split(",") if item.strip()]
    return ", ".join(sorted(set(parts)))


def load_reference_map(path: str, column: str) -> dict:
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if column not in df.columns:
        return {}
    values = [str(item).strip() for item in df[column].dropna().tolist()]
    return {value.lower(): value for value in values}


def load_main_dataset() -> pd.DataFrame:
    if not os.path.exists(INPUT_MAIN):
        raise FileNotFoundError(f"Missing main dataset: {INPUT_MAIN}")
    return pd.read_csv(INPUT_MAIN)


def maybe_merge_personality(df: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(OPTIONAL_PERSONALITY):
        return df

    personality_df = pd.read_csv(OPTIONAL_PERSONALITY)
    if "user_id" not in personality_df.columns or "user_id" not in df.columns:
        return df

    # Only bring in extra columns not already in the main dataset.
    extra_cols = [col for col in personality_df.columns if col != "user_id" and col not in df.columns]
    if not extra_cols:
        return df

    return df.merge(personality_df[["user_id"] + extra_cols], on="user_id", how="left")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    passion_map = load_reference_map(PASSIONS_PATH, "passion")
    hobby_map = load_reference_map(HOBBIES_PATH, "hobby")
    skill_map = load_reference_map(SKILLS_PATH, "skill_name")
    career_map = load_reference_map(CAREERS_PATH, "career_title")

    if "education_level" in df.columns:
        df["education_level"] = df["education_level"].apply(normalize_education)

    if "passion" in df.columns and passion_map:
        df["passion"] = df["passion"].apply(
            lambda value: passion_map.get(str(value).strip().lower())
        )

    if "hobby" in df.columns and hobby_map:
        df["hobby"] = df["hobby"].apply(
            lambda value: hobby_map.get(str(value).strip().lower())
        )

    if "skills" in df.columns:
        def normalize_skill_list(value: str) -> str:
            if not isinstance(value, str):
                return value
            parts = [item.strip() for item in value.split(",") if item.strip()]
            if skill_map:
                normalized = [skill_map.get(part.lower()) for part in parts]
                normalized = [item for item in normalized if item]
            else:
                normalized = parts
            return ", ".join(sorted(set(normalized)))

        df["skills"] = df["skills"].apply(normalize_skill_list)

    if "career_title" in df.columns and career_map:
        df["career_title"] = df["career_title"].apply(
            lambda value: career_map.get(str(value).strip().lower())
        )

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows without the target label or essential features.
    required_cols = [
        "career_title",
        "skills",
        "education_level",
        "age",
        "passion",
        "hobby",
    ]
    existing_required = [col for col in required_cols if col in df.columns]
    if existing_required:
        df = df.dropna(subset=existing_required)

    df = df.drop_duplicates()
    return df


def main() -> None:
    df = load_main_dataset()
    df = maybe_merge_personality(df)
    df = clean_dataset(df)

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(OUTPUT_CLEAN, index=False)

    print(f"✅ Clean training dataset saved to: {OUTPUT_CLEAN}")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()
