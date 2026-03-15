from fastapi import FastAPI, HTTPException
from schemas.student_schema import StudentData
from services.ml_service import predict_cluster
from services.llm_service import generate_llm_analysis

app = FastAPI(title="Analyze Student Behavior API", version="1.0")

@app.post("/analyze-student")
async def analyze_student(data: StudentData):
    try:

        input_dict = data.model_dump()

        persona_name = predict_cluster(input_dict)

        llm_json = generate_llm_analysis(
            input_dict=input_dict,
            persona_name=persona_name
        )

        return {
            "status": "success",
            "data": llm_json
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))