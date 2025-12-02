# from dash import html, dcc, dash, Input, Output
# from flask import request
# from typing import Dict

# # from dash.dependencies import Output, Input
# from FlaskHeaderChecker.FlaskHeaderChecker import FlaskHeaderChecker


# # Initialize Dash app
# app = dash.Dash(__name__)
# utility_name = "EPRI"
# server = app.server  # Flask server instance

# header_checker = FlaskHeaderChecker(utility_name=utility_name)

# # Layout with a placeholder for headers
# app.layout = html.Div(
#     [
#         html.H1("Request Headers"),
#         html.Div(id="headers-output"),
#         dcc.Interval(
#             id="interval", interval=1000, n_intervals=0, max_intervals=1
#         ),  # triggers once
#     ]
# )


# # Callback to display headers when the app loads
# @app.callback(Output("headers-output", "children"), Input("interval", "n_intervals"))
# def display_headers(n):
#     # Access Flask request headers
#     headers_list = [f"{header}: {value}" for header, value in request.headers.items()]
#     header_verified = header_checker.verify(dict(request.headers))
#     headers_list.append(f"Header Verified: {header_verified}")

#     return html.Pre("\n".join(headers_list))


# if __name__ == "__main__":
#     app.run(debug=True)


from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from celery.result import AsyncResult
import time
from typing import Optional

app = FastAPI(title="FastAPI + Celery + Redis")

# Import celery worker
from celery_worker import celery_app, long_running_task


@app.get("/")
def read_root():
    return {"message": "FastAPI + Celery + Redis API"}


@app.post("/tasks/long-running")
def create_long_task(duration: int = 30):
    """
    Create a long-running task
    Args:
        duration: How long the task should run (in seconds)
    """
    task = long_running_task.delay(duration)
    return JSONResponse(
        {
            "task_id": task.id,
            "status": "Task submitted",
            "message": f"Task will run for {duration} seconds",
        }
    )


@app.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    """
    Get the status of a task
    """
    task_result = AsyncResult(task_id, app=celery_app)

    result = {
        "task_id": task_id,
        "status": task_result.state,
    }

    if task_result.state == "PENDING":
        result["message"] = "Task is waiting to be processed"
    elif task_result.state == "STARTED":
        result["message"] = "Task is currently running"
        result["progress"] = (
            task_result.info.get("progress", 0) if task_result.info else 0
        )
    elif task_result.state == "SUCCESS":
        result["message"] = "Task completed successfully"
        result["result"] = task_result.result
    elif task_result.state == "FAILURE":
        result["message"] = "Task failed"
        result["error"] = str(task_result.info)
    elif task_result.state == "PROGRESS":
        result["message"] = "Task in progress"
        result["progress"] = task_result.info.get("progress", 0)
        result["current_step"] = task_result.info.get("current", 0)
        result["total_steps"] = task_result.info.get("total", 0)

    return JSONResponse(result)


@app.delete("/tasks/{task_id}")
def cancel_task(task_id: str):
    """
    Cancel a running task
    """
    task_result = AsyncResult(task_id, app=celery_app)
    task_result.revoke(terminate=True)

    return JSONResponse({"task_id": task_id, "message": "Task cancellation requested"})
