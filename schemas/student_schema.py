from pydantic import BaseModel

class StudentData(BaseModel):
    total_active_days: float
    weekend_ratio: float
    avg_speed_ratio: float
    preferred_study_hour: float
    avg_project_score: float
    avg_project_difficulty: float
    avg_procrastination_days: float
    total_projects_completed: float
    avg_attempts_per_project: float
    avg_exam_score: float
    avg_exam_difficulty: float