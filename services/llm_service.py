import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from prompts.student_prompt import build_llm_payload, SYSTEM_PROMPT

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

def generate_llm_analysis(input_dict, persona_name):

    llm_input_payload = build_llm_payload(input_dict, persona_name)

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": json.dumps(SYSTEM_PROMPT)
            },
            {
                "role": "user",
                "content": json.dumps(llm_input_payload)
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=800,
        temperature=0.3,
        timeout=15
    )

    llm_text = response.choices[0].message.content

    llm_json = json.loads(llm_text)

    return llm_json