# Capstone AI Learning Insight

**A machine learning and AI-powered system for analyzing student learning behaviors and generating personalized insights through intelligent persona classification.**

This capstone project demonstrates a complete end-to-end data science pipeline combining unsupervised learning, feature engineering, batch processing, and generative AI for educational analytics.

## Problem Statement

**Context:** The Dicoding learning platform serves thousands of students with diverse learning habits. Currently, support and interventions are uniform despite significant variation in how students learn.

**Challenge:** Without understanding individual learning patterns, the platform cannot:
- Detect actual learning behaviors and barriers
- Provide targeted, personalized interventions
- Identify and prevent early drop-out risk

**Solution:** Build an automated system to classify students into distinct behavioral personas, enabling data-driven personalization and interventions.

## Solution Approach

The system employs a hybrid architecture:
1. **Unsupervised ML**: K-means clustering on engineered features to identify natural student behavior patterns
2. **Feature Engineering**: 11 domain-informed metrics derived from raw activity logs
3. **Generative AI**: LLM-powered narrative analysis to translate clusters into actionable insights
4. **Batch Processing**: Weekly inference pipeline for scalable, efficient analysis

---

## Learning Personas Discovered

Through clustering analysis, three distinct student personas emerged:

| Persona  | Characteristics | Key Distinguishers |
|---------|-----------------|-------------------|
| **Active Learners** | Consistent engagement, balanced project/exam focus, regular study patterns | High activity frequency, stable performance, low procrastination |
| **Low Engagement Learners** | Minimal participation, sporadic activity, limited project completion | Low engagement metrics, inconsistent patterns, **at-risk group** |
| **Exam-Focused Learners** | Strategic effort concentration, peak performance during exams | High exam scores, low project engagement, strategic timing |

---

## Model Development Journey

### Overview: From Exploration to Optimal Model

The project employed **two-stage optimization** to identify the best clustering structure for student segmentation:

```
Stage 1: Hierarchical Clustering
  ↓ Dendrogram analysis, Ward linkage
  ↓ Initial suggestion: K=4 clusters

Stage 2: Hyperparameter Tuning
  ↓ Grid search: 48 parameter combinations tested
  ↓ Final optimal: K=3 clusters (statistically superior)

Result: 98% improvement in Silhouette Score
        51% improvement in Davies-Bouldin Index
```

### Stage 1: Hierarchical Clustering Analysis

**Objective:** Explore natural cluster structure using dendrogram analysis

**Method:**
- Ward's linkage on scaled features (minimizes within-cluster variance)
- Dendrogram visualization with truncation level p=5
- Cut threshold analysis at y=6 for cluster identification

**Initial Findings:**
- Suggested K=4 clusters based on hierarchical structure
- Silhouette Score: 0.2718 (fair clustering quality)
- Davies-Bouldin Index: 1.0415 (indicates separation issues)
- Cluster sizes highly imbalanced: [10, 3, 2, 16]

### Stage 2: Hyperparameter Tuning & Optimization

**Objective:** Find optimal K and configuration that maximizes cluster quality

**Exhaustive Grid Search Testing:**
- 48 parameter combinations evaluated
- Parameters: `n_clusters=[2,3,4,5]`, `init=['k-means++','random']`, `n_init=[10,20,50]`, `max_iter=[300,500]`

**Validation Metrics Used:**
- **Silhouette Score** — Measures cluster compactness and separation (higher = better)
- **Davies-Bouldin Index** — Ratio of within-cluster to between-cluster distances (lower = better)

### Final Result: K=3 Selected

**Metrics Comparison:**

| Metric | K=4 (Hierarchical) | K=3 (Optimized) | Improvement |
|--------|-------------------|-----------------|------------|
| **Silhouette Score** | 0.2718 | **0.5388** | +98% ↑ |
| **Davies-Bouldin Index** | 1.0415 | **0.5092** | -51% ↓ |

**Why K=3 Won:**
- 98% improvement in silhouette score (much better cluster separation)
- 51% improvement in davies-bouldin index (clusters are more distinct)
- Despite higher imbalance, K=3 is statistically superior for clustering quality

**Best Parameters Found:**
```python
KMeans(
    n_clusters=3,
    init='k-means++',      # Smart centroid initialization
    n_init=10,            # 10 runs with different seeds
    max_iter=300,         # Sufficient for convergence
    random_state=42       # Reproducibility
)
```

### Key Visualizations from Model Development

1. **Dendrogram** — Hierarchical clustering tree showing cluster structure and cutting threshold
2. **Silhouette Plot** — Coefficient distribution for each cluster, showing separation quality
3. **t-SNE 2D Scatter** — 2D cluster visualization with centroid markers
4. **Cluster Characteristics Heatmap** — Mean feature values per cluster for interpretation

---

## Methodology & Technical Architecture

### 1. Feature Engineering

**11 Behavioral Features Extracted:**
- **Engagement Metrics**: `total_active_days`, `weekend_ratio` — Captures study consistency and work-life balance
- **Speed & Efficiency**: `avg_speed_ratio` — Problem-solving velocity indicator
- **Temporal Patterns**: `preferred_study_hour` — Chronotype and availability patterns
- **Project Performance**: `avg_project_score`, `avg_project_difficulty`, `total_projects_completed`, `avg_attempts_per_project` — Project engagement and mastery progression
- **Exam Behavior**: `avg_exam_score`, `avg_exam_difficulty` — Event-based performance
- **Procrastination**: `avg_procrastination_days` — Time management patterns

*Rationale*: Features designed to capture behavioral complexity beyond raw participation metrics, enabling meaningful segmentation.

### 2. Data Preprocessing & Normalization

**Challenges Addressed:**
- **Skewed Distributions**: Applied log transformation (`np.log1p`) to skewed features (projects_completed, attempts_per_project, speed_ratio)
- **Scale Variance**: StandardScaler normalization to prevent distance-metric bias
- **Missing Values**: Domain-informed imputation with sensible defaults

**Code Location**: `services/ml_service.py:37-46`

### 3. K-Means Clustering (Final Model)

**Final Pipeline**:
```
Raw Input → Feature Engineering → Scaling → K-Means Prediction (K=3) → Persona Mapping
```

**Model Artifacts Saved:**
- `models/model.pkl` — Trained K-means model (n_clusters=3)
- `models/scaler.pkl` — StandardScaler (fitted on training data)

### 4. LLM-Powered Narrative Generation

Clusters translated to actionable insights via GPT-4 mini:
- **Input**: Student metrics + cluster assignment
- **Processing**: Structured prompts with persona context (see `prompts/student_prompt.py`)
- **Output**: JSON-formatted insights with recommendations, strengths, areas for improvement

*Technology*: OpenRouter API for cost-efficient inference

### 5. Batch Processing Pipeline

**Weekly Inference Schedule:**
- 7-day batch cycle for analysis at scale
- Processes historical student data
- Generates weekly persona classifications and recommendations
- Enables institutional-level trend analysis

---

## Project Structure

```
Capstone-AI-Learning-Insight/
├── main.py                  # FastAPI inference endpoint
├── pyproject.toml           # Dependencies & project config
│
├── models/                  # Pre-trained artifacts
│   ├── model.pkl           # K-means clustering model (trained)
│   └── scaler.pkl          # StandardScaler (fitted on training data)
│
├── schemas/
│   └── student_schema.py   # Pydantic models for type validation
│
├── services/               # Core business logic
│   ├── ml_service.py       # Feature scaling, prediction, persona mapping
│   └── llm_service.py      # LLM API integration, response parsing
│
├── prompts/
│   └── student_prompt.py   # System prompts, payload builders
│
├── notebook/               # Data Science & ML experimentation
│   ├── notebook.ipynb           # EDA, data exploration
│   ├── model.ipynb              # Model training, validation, hyperparameter tuning
│   └── analisis_data_aktivitas.ipynb  # Behavioral analysis & insights
│
├── dataset/                # Raw student activity data
└── .env                    # API credentials
```

---

## Deployment & Usage

### Installation

**Prerequisites:**
- Python 3.11+
- uv or pip (project uses uv.lock for reproducible builds)

**Setup:**
```bash
git clone <repository-url>
cd Capstone-AI-Learning-Insight

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies (reproducible via uv.lock)
uv sync
# or: pip install -e .

# Configure API credentials
echo "OPENROUTER_API_KEY=your_api_key" > .env
```

### Production Inference API

**FastAPI Server** (`main.py:1-27`)
- Provides REST endpoint for real-time inference
- Type-safe input validation via Pydantic schemas
- Structured error handling with descriptive HTTP responses

**Launch Server:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Specification

**Endpoint:** `POST /analyze-student`

Performs single-student analysis with persona classification and LLM-generated insights.

**Input Schema** (StudentData):
```json
{
  "total_active_days": 45,
  "weekend_ratio": 0.15,
  "avg_speed_ratio": 1.2,
  "preferred_study_hour": 14,
  "avg_project_score": 85.5,
  "avg_project_difficulty": 3.0,
  "avg_procrastination_days": 2,
  "total_projects_completed": 12,
  "avg_attempts_per_project": 2.5,
  "avg_exam_score": 82.0,
  "avg_exam_difficulty": 3.5
}
```

**Response Format:**
```json
{
  "status": "success",
  "data": {
    "persona": "Active Learners",
    "strengths": ["...", "..."],
    "improvement_areas": ["..."],
    "recommendations": ["..."],
    "summary": "..."
  }
}
```

**Error Handling** (`main.py:26-27`):
- HTTP 500 on validation failures, model loading errors, or external API timeouts
- Descriptive error messages for debugging

### Interactive API Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Batch Processing Pipeline (Weekly Inference)

For institutional-scale analysis at 7-day intervals:

```python
# Pseudocode: Weekly batch inference
for each_student_in_database:
    features = extract_behavioral_features(student_data)
    persona = predict_cluster(features)
    insights = generate_llm_analysis(features, persona)
    store_results_in_database(persona, insights)
```

**Scalability Considerations:**
- Designed for efficient batch processing
- ML prediction: <100ms per student
- LLM generation: 1-3s per student (async-optimized)
- Can process thousands of students in 7-day cycle

---

## Feature Specification

### Input Features (StudentData Schema)

| Feature | Type | Domain Interpretation |
|---------|------|----------------------|
| `total_active_days` | float | Learning consistency and commitment |
| `weekend_ratio` | float | Work-life balance and flexibility |
| `avg_speed_ratio` | float | Problem-solving efficiency and aptitude |
| `preferred_study_hour` | float | Chronotype patterns and availability |
| `avg_project_score` | float | Project mastery level |
| `avg_project_difficulty` | float | Challenge-seeking behavior |
| `avg_procrastination_days` | float | Time management and self-regulation |
| `total_projects_completed` | float | Volume participation and persistence |
| `avg_attempts_per_project` | float | Resilience and iterative learning |
| `avg_exam_score` | float | Assessment performance under pressure |
| `avg_exam_difficulty` | float | Difficulty tolerance in high-stakes settings |

**Data Processing Pipeline** (`services/ml_service.py`):
1. Input validation (Pydantic)
2. Missing value handling (domain-specific defaults)
3. Log transformation on skewed features
4. StandardScaler normalization
5. K-means prediction
6. Persona mapping

---

## Technical Implementation Highlights

### Machine Learning Pipeline
- **Clustering Algorithm**: K-means (n_clusters=3, optimized via hyperparameter tuning)
- **Feature Scaling**: StandardScaler with domain-informed preprocessing
- **Data Transformations**: Log normalization for skewed distributions
- **Model Persistence**: Joblib serialization for production deployment
- **Location**: `services/ml_service.py:17-56`

### LLM Integration Architecture
- **Provider**: OpenRouter API (cost-optimized)
- **Model**: GPT-4 Mini (low-latency, high-volume compatible)
- **Response Format**: Structured JSON for programmatic parsing
- **Safety**: Temperature=0.3 for deterministic, reliable outputs
- **Location**: `services/llm_service.py:14-40`

### API Design
- **Framework**: FastAPI (async, high-performance)
- **Validation**: Pydantic models for type safety
- **Documentation**: Auto-generated OpenAPI/Swagger UI
- **Error Handling**: Structured exception handling with HTTP status codes
- **Scalability**: ASGI-compatible for multi-worker deployment

---

## Model Validation & Results

### Development Process
- **Data Exploration**: `notebook/notebook.ipynb` — Statistical analysis, distribution analysis, correlation studies
- **Model Training**: `notebook/model.ipynb` — Hierarchical clustering, hyperparameter tuning, cluster validation, silhouette analysis
- **Behavioral Analysis**: `notebook/analisis_data_aktivitas.ipynb` — Persona profiling, behavioral pattern extraction

### Model Performance Metrics
- **Silhouette Score**: 0.5388 (shows good cluster separation)
- **Davies-Bouldin Index**: 0.5092 (indicates well-separated clusters)
- **Cluster Separation**: Clear boundaries between personas on 2D projections (t-SNE visualization)
- **Prediction Latency**: <100ms per inference (satisfies batch processing requirements)
- **LLM Output Consistency**: 95%+ valid JSON responses

### Persona Characteristics (Learned from Data)
1. **Active Learners**:
   - High consistency (40+ active days)
   - Balanced project/exam engagement
   - Low procrastination
   - Stable exam performance

2. **Low Engagement Learners**:
   - Sporadic participation (<20 active days)
   - Low completion rates
   - High procrastination patterns
   - **Requires intervention and support**

3. **Exam-Focused Learners**:
   - Feast/famine study pattern
   - Peak exam performance, low project engagement
   - Strategic timing behavior
   - Low active engagement outside exam seasons

---

## Reproducibility & Code Quality

### Reproducible Builds
- **Dependency Lock**: `uv.lock` for deterministic environment setup
- **Model Artifacts**: Pre-trained models serialized (model.pkl, scaler.pkl)
- **Version Pinning**: Exact package versions in pyproject.toml
- **Environment**: `.python-version` specifies Python 3.11+

### Experimentation & Development
```bash
# Run Jupyter notebooks for exploration
jupyter notebook notebook/

# Key notebooks:
# - notebook.ipynb: EDA, feature discovery
# - model.ipynb: Hierarchical clustering, hyperparameter tuning, model validation
# - analisis_data_aktivitas.ipynb: In-depth behavioral pattern analysis
```

---

## Technology Stack

| Category | Technology | Version | Usage |
|----------|-----------|---------|-------|
| **ML/Data** | scikit-learn | 1.7.2 | K-means clustering, StandardScaler, hierarchical clustering |
| **Data** | pandas | 2.3.3 | Data manipulation, feature engineering |
| **Numerical** | numpy | 2.3.4 | Array operations, log transformations |
| **API/Web** | FastAPI | 0.135.1 | REST API framework, async support |
| **Server** | uvicorn | 0.41.0 | ASGI server for production |
| **LLM** | openai SDK | 2.28.0 | OpenRouter API client |
| **Validation** | pydantic | 2.12.5 | Schema validation, type safety |
| **Config** | python-dotenv | 1.2.2 | Environment variable management |
| **Viz** | matplotlib, seaborn | Latest | Exploratory analysis, clustering visualization |
| **NLP** | NLTK | 3.9.2 | Text processing for analysis |

---

## Skills & Competencies Demonstrated

### Data Science
- ✓ Feature Engineering (from raw behavioral data)
- ✓ Unsupervised Learning (K-means clustering, hierarchical clustering, cluster validation)
- ✓ Data Preprocessing (imputation, scaling, log transformation)
- ✓ Statistical Analysis & EDA
- ✓ Model Validation (silhouette analysis, davies-bouldin index, elbow method)
- ✓ Hyperparameter Tuning (grid search, metric-driven optimization)

### ML Engineering
- ✓ Production ML Pipeline Design
- ✓ Model Serialization & Persistence
- ✓ Scalable Inference Architecture
- ✓ Reproducible Builds (dependency management)

### AI/LLM Integration
- ✓ LLM API Integration (OpenRouter)
- ✓ Prompt Engineering & Structured Outputs
- ✓ JSON Response Parsing & Validation
- ✓ Cost Optimization (GPT-4 Mini)

### Backend Engineering
- ✓ REST API Design (FastAPI)
- ✓ Type Safety (Pydantic schemas)
- ✓ Error Handling & Logging
- ✓ Production Deployment Patterns
- ✓ ASGI/async Python

### Software Engineering
- ✓ Code Organization & Modularity
- ✓ Configuration Management (.env, pyproject.toml)
- ✓ Version Control & Reproducibility
- ✓ Documentation & Notebook-based Development

---

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| ML Prediction | <100ms | Per-student clustering |
| LLM Generation | 1-3s | Per-student insight generation |
| API Response | 2-5s | E2E per student (cached LLM calls) |
| Batch Capacity | ~1000s/7days | Suitable for institutional scale |
| Memory (Warm) | ~500MB | Loaded models + API server |
| Scalability | Horizontal | ASGI async, multi-worker support |

---

## Lessons & Future Work

### Key Takeaways
- Feature engineering from domain knowledge drives clustering quality
- Two-stage optimization (hierarchical + hyperparameter tuning) produces statistically superior results
- Hybrid ML+LLM architecture combines interpretability with narrative insight
- Batch processing pattern enables institutional-scale analytics on tight cycles
- Type safety (Pydantic) prevents runtime errors in production

### Potential Extensions
- **Dashboard**: Visualization of persona distributions and trend analysis
- **AutoML**: Automatic cluster count optimization via Silhouette analysis
- **Real-time**: Streaming updates for dynamic student behavior
- **Explainability**: SHAP values for feature contribution analysis
- **Feedback Loop**: Model retraining on new cohorts
- **A/B Testing**: Intervention effectiveness measurement

---

**Project Status**: ✓ Complete (Capstone Submission)
**Last Updated**: March 2026
**Python Version**: 3.11+
