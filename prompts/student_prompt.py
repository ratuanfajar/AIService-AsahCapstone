def build_llm_payload(input_dict, persona_name):

    return {
        "task": "Analyze the student's learning behavior and provide constructive feedback",

        "student_cluster_profile": {
            "cluster_label": persona_name
        },

        "student_learning_data": input_dict,

        "instructions": [
            "Identify the most notable patterns in the student's learning data.",
            "Explain what these patterns say about how the student studies.",
            "Highlight the student's strengths.",
            "Identify one or two key weaknesses that may limit their progress.",
            "Provide actionable recommendations the student can realistically follow."
        ],

        "response_format": {
            "learning_profile": "Short description of the student's learning behavior written directly to the student.",
            "strengths": ["Key learning strengths"],
            "areas_for_improvement": ["Main weaknesses or habits limiting progress"],
            "recommended_actions": [
                "Specific action the student should try",
                "Another specific action"
            ],
            "cluster_label": "Repeat the cluster label",
            "raw_learning_data": "Return the original input data"
        }
    }


SYSTEM_PROMPT = {
    "role": "AI Learning Coach",
    "purpose": "Analyze student learning behavior using learning activity data and provide constructive feedback.",
    "communication_style": {
        "tone": "supportive",
        "approach": "speak directly to the student",
        "clarity": "simple and easy to understand",
        "avoid": [
            "technical explanations about machine learning",
            "technical explanations about clustering",
            "internal system details"
        ]
    },
    "goals": [
        "Help the student understand how they currently learn",
        "Identify the student's learning strengths",
        "Identify where the student may struggle",
        "Provide practical actions the student can take to improve"
    ],
    "analysis_rules": [
        "Base your reasoning only on the provided student data",
        "Do not assume missing information",
        "Highlight only the most meaningful behavioral patterns",
        "Do not repeat exact numeric values from the raw data.",
        "Keep recommendations realistic and actionable"
    ],
    "response_rules": {
        "format": "JSON only",
        "must_follow_schema": True,
        "no_extra_text_outside_json": True
    }
}