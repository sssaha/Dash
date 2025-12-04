from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from celery.result import AsyncResult
import time
import logging
import sys
from typing import Optional

# Configure logging for Azure App Service
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

app = FastAPI(title="FastAPI + Celery + Azure Redis")

# THIS IS WHERE REDIS CONNECTION HAPPENS:
# When you import celery_worker, it creates the celery_app instance
# which automatically connects to Redis using the connection string
# defined in celery_worker.py
try:
    from celery_worker import celery_app, long_running_task

    logger.info("Successfully imported celery_worker")
    logger.info(f"Celery broker: {celery_app.conf.broker_url}")
    logger.info(f"Celery backend: {celery_app.conf.result_backend}")
except Exception as e:
    logger.error(f"Failed to import celery_worker: {str(e)}")
    raise


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "path": str(request.url),
        },
    )


# Validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.get("/debug/celery")
def debug_celery():
    """Debug endpoint to check Celery configuration and worker status"""
    try:
        # Force reconnection before inspection
        celery_app.connection().ensure_connection(max_retries=3)

        # Get active workers with timeout
        inspect = celery_app.control.inspect(timeout=5.0)
        active_workers = inspect.active()
        registered_tasks = inspect.registered()
        stats = inspect.stats()

        return {
            "celery_config": {
                "broker": celery_app.conf.broker_url.split("@")[1]
                if "@" in celery_app.conf.broker_url
                else "hidden",
                "backend": celery_app.conf.result_backend.split("@")[1]
                if "@" in celery_app.conf.result_backend
                else "hidden",
            },
            "workers": {
                "active": active_workers or "No workers found",
                "registered_tasks": registered_tasks or "No workers found",
                "stats": stats or "No workers found",
            },
            "task_info": {
                "registered_task_name": "long_running_task",
                "task_module": long_running_task.__module__,
            },
        }
    except Exception as e:
        logger.error(f"Debug endpoint error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={
                "error": str(e),
                "type": type(e).__name__,
                "message": "Failed to inspect Celery workers. This may indicate workers are not running or Redis connection issues.",
            },
        )


@app.get("/")
def read_root():
    logger.info("Root endpoint called")
    return {"message": "FastAPI + Celery + Azure Redis API", "status": "running"}


@app.get("/health")
def health_check():
    """Health check endpoint for Azure App Service"""
    try:
        # Force fresh connection and ping
        with celery_app.connection_or_acquire() as conn:
            conn.ensure_connection(max_retries=3)
            conn.default_channel.client.ping()

        logger.info("Health check passed - Redis connected")
        return {
            "status": "healthy",
            "redis": "connected",
            "celery": "configured",
            "broker": celery_app.conf.broker_url.split("@")[1]
            if "@" in celery_app.conf.broker_url
            else "configured",
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503, content={"status": "unhealthy", "error": str(e)}
        )


@app.post("/tasks/long-running")
def create_long_task(duration: int = 30):
    """
    Create a long-running task
    Args:
        duration: How long the task should run (in seconds)
    """
    try:
        logger.info(f"Creating task with duration: {duration}")
        # THIS IS WHERE FASTAPI SENDS DATA TO REDIS:
        # .delay() serializes the task and sends it to Redis queue
        # Redis stores it in a list that Celery workers monitor
        task = long_running_task.delay(duration)
        logger.info(f"Task created with ID: {task.id}")
        return JSONResponse(
            {
                "task_id": task.id,
                "status": "Task submitted",
                "message": f"Task will run for {duration} seconds",
            }
        )
    except Exception as e:
        logger.error(f"Error creating task: {str(e)}", exc_info=True)
        raise


@app.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    """
    Get the status of a task
    """
    try:
        logger.info(f"Checking status for task: {task_id}")
        # THIS IS WHERE FASTAPI READS FROM REDIS:
        # AsyncResult queries Redis backend for task status
        # It retrieves the task state and result from Redis
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
            logger.error(f"Task {task_id} failed: {task_result.info}")
        elif task_result.state == "PROGRESS":
            result["message"] = "Task in progress"
            result["progress"] = task_result.info.get("progress", 0)
            result["current_step"] = task_result.info.get("current", 0)
            result["total_steps"] = task_result.info.get("total", 0)

        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Error checking task status: {str(e)}", exc_info=True)
        raise


@app.delete("/tasks/{task_id}")
def cancel_task(task_id: str):
    """
    Cancel a running task
    """
    try:
        logger.info(f"Cancelling task: {task_id}")
        # THIS SENDS A REVOKE COMMAND TO REDIS:
        # The revoke message is stored in Redis
        # Celery workers check for revoke commands before processing
        task_result = AsyncResult(task_id, app=celery_app)
        task_result.revoke(terminate=True)

        return JSONResponse(
            {"task_id": task_id, "message": "Task cancellation requested"}
        )
    except Exception as e:
        logger.error(f"Error cancelling task: {str(e)}", exc_info=True)
        raise
