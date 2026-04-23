# import json
# import re
# from src.functions.utils.cloudstorage import GoogleCloudStorage

import os

from time import time
from pathlib import Path
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google import genai

from src.functions.core.hydegenerator import HydeGenerator
import logging

# Disable verbose logs from LLM client
logging.getLogger("src.functions.utils.llm_client").setLevel(logging.ERROR)

# Optional: reduce Google SDK logs
logging.getLogger("google").setLevel(logging.WARNING)

app = FastAPI(
    title="Feed recommentdation HyDE",
    version="1.1.0",
    description=(
        "Feed recommendation HyDE part"
        "<br>"
        f"Last time Update : 2026-04-23 14:54"
        "<br>"
        "Repo : https://github.com/TunKedsaro/feed_recommend_hyde"
    ),
    contact={
        "name": "Tun Kedsaro",
        "email": "tun.k@terradigitalventures.com",
        
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
### ----------      setting      ---------- ###
config_path = Path(__file__).resolve().parent / "parameters" / "parameters.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

bucket_name = config["cloudstorage"]["bucket"]

### ---------- Health & Metadata ---------- ###
### ----------     API:0.0       ---------- ###
@app.get(
    "/", 
    tags=["Health & Metadata"],
    description="API 0.0 : Service root status check"
)
def root_status():
    return {
        "response":"ok"
    }

### ---------- Health & Metadata ---------- ###
### ----------     API:1.1       ---------- ###
@app.get(
    "/health/", 
    tags=["Health & Metadata"],
    description="API 1.1 : Basic service health check"
)
def health_check():
    start_time  = time()
    finish_time = time()
    process_time = finish_time - start_time
    return {
        "status": "ok", 
        "service": "FastAPI",
        "response_time" : f"{process_time:.5f} s"
        }


### ----------     API:1.2       ---------- ###
@app.get(
    "/health/gemini", 
    tags=["Health & Metadata"],
    description="API 1.2 : Gemini connectivity and latency check"
)
def gemini_health_check():
    start_time  = time()
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="ping"
    )
    finish_time = time()
    process_time = finish_time - start_time
    return {
        "status": "ok",
        "reply": resp.text,
        "latencyresponse_time_sec": f"{process_time:.5f} s"
    }


### ----------     API:1.3       ---------- ###
from google.cloud import bigquery
def get_user_events(user_id: str):
    client = bigquery.Client()
    query = """
        SELECT *
        FROM `poc-piloturl-nonprod.gold_layer.students`
        WHERE student_id = @user_id
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
        ]
    )
    rows = client.query(query, job_config=job_config)
    return [dict(row) for row in rows]

@app.get(
    "/health/bigquery", 
    tags=["Health & Metadata"],
    description="API 1.3 : BigQuery connectivity test query"
)
def bigquery_health_check():
    start_time  = time()
    finish_time = time()
    process_time = finish_time - start_time
    return {
        "status": "ok", 
        "body": get_user_events("stu_p001"),
        "response_time" : f"{process_time:.5f} s"
        }

### ----------   Hyde Generator  ---------- ###
hg = HydeGenerator(
    bucket_name = bucket_name,
    verbose     = 0
)
### ----------     API:2.1       ---------- ###
@app.post(
    "/hyde/students/{student_id}", 
    tags=["Hyde Generator"],
    description="API 2.1 : Generate HyDE bundle for a single student"
)
def generate_student_recommendation(student_id):
    status = hg.single_hyde_generator2(student_id=student_id)
    return {
        "student_id":student_id,
        "response"  :status
    }

### ----------     API:2.2       ---------- ###
@app.get(
    "/hyde/students/sequential", 
    tags=["Hyde Generator"],
    description="API 2.2 : Generate HyDE bundle every students in bigquery"
)
def sequential_of_single_hyde_generator():
    report_each_student = hg.sequential_of_single_student_generator()
    return {
        "report_each_student":report_each_student
    }

