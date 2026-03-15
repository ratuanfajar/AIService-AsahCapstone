import pandas as pd
import numpy as np
import joblib

try:
    scaler = joblib.load('models/scaler.pkl')
    kmeans_model = joblib.load('models/model.pkl')
except Exception as e:
    print(f"error memuat model: {e}")

CLUSTER_NAMES = {
    0: "Active Learners",
    1: "Low Engagement Learners",
    2: "Exam-Focused Learners"
}

def predict_cluster(input_dict):

    df_input = pd.DataFrame([input_dict])

    fill_values = {
        'total_active_days': 0,
        'weekend_ratio': 0.0,
        'avg_speed_ratio': 1.0,
        'preferred_study_hour': 12.0,
        'avg_project_score': 0.0,
        'avg_project_difficulty': 1.0,
        'avg_procrastination_days': 0.0,
        'total_projects_completed': 0,
        'avg_attempts_per_project': 0,
        'avg_exam_score': 0.0,
        'avg_exam_difficulty': 1.0
    }

    df_processed = df_input.fillna(value=fill_values)

    skewed_cols = [
        'total_projects_completed',
        'avg_attempts_per_project',
        'avg_speed_ratio'
    ]

    for col in skewed_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].clip(lower=0)
            df_processed[col] = np.log1p(df_processed[col])

    scaled_array = scaler.transform(df_processed)

    df_scaled = pd.DataFrame(scaled_array, columns=df_processed.columns)

    cluster_id = int(kmeans_model.predict(df_scaled)[0])

    persona_name = CLUSTER_NAMES.get(cluster_id, "unknown")

    return persona_name