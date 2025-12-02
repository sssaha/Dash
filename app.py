
# app.py
from fastapi import FastAPI
from celery import Celery
import time
import os

app = FastAPI()

# Get Redis connection from environment variables
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery = Celery(
    'tasks',
    broker=REDIS_URL,
    backend=REDIS_URL
)

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI with Celery!"}

@app.post("/process-task/")
def process_task():
    task = long_running_task.delay()
    return {"task_id": task.id, "status": "Task submitted"}

@celery.task
def long_running_task():
    time.sleep(10)  # Simulate long-running work
    return "Task completed!"
