from celery import Celery
import time
import ssl, os

REDIS_URL = os.getenv("REDIS_URL", "rediss://:")
# Configure Celery
celery_app = Celery("worker", broker=REDIS_URL, backend=REDIS_URL)

celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    broker_use_ssl={
        "ssl_cert_reqs": ssl.CERT_REQUIRED,
        "ssl_ca_certs": None,  # Use system CA certs
    },
    redis_backend_use_ssl={
        "ssl_cert_reqs": ssl.CERT_REQUIRED,
        "ssl_ca_certs": None,  # Use system CA certs
    },
    # Connection settings
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=10,
)


@celery_app.task(bind=True, name="long_running_task")
def long_running_task(self, duration: int):
    """
    Simulate a long-running task with progress updates
    """
    steps = 10
    step_duration = duration / steps

    for i in range(steps):
        time.sleep(step_duration)

        # Update progress
        self.update_state(
            state="PROGRESS",
            meta={
                "current": i + 1,
                "total": steps,
                "progress": int((i + 1) / steps * 100),
            },
        )

    return {
        "status": "completed",
        "duration": duration,
        "message": f"Task completed after {duration} seconds",
    }
