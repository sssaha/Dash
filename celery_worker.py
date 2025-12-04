from celery import Celery
import time
import os
import ssl
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

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
    broker_pool_limit=10,
    broker_transport_options={
        "visibility_timeout": 3600,
        "retry_on_timeout": True,
        "socket_keepalive": True,
        "socket_keepalive_options": {
            1: 1,  # TCP_KEEPIDLE
            2: 1,  # TCP_KEEPINTVL
            3: 5,  # TCP_KEEPCNT
        },
        "health_check_interval": 30,
    },
    result_backend_transport_options={
        "retry_on_timeout": True,
        "socket_keepalive": True,
        "health_check_interval": 30,
    },
)

logger.info("Celery configured successfully")


@celery_app.task(bind=True, name="long_running_task")
def long_running_task(self, duration: int):
    """
    Simulate a long-running task with progress updates
    """
    logger.info(f"Task {self.request.id} started with duration {duration}s")

    try:
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
            logger.info(
                f"Task {self.request.id} progress: {int((i + 1) / steps * 100)}%"
            )

        result = {
            "status": "completed",
            "duration": duration,
            "message": f"Task completed after {duration} seconds",
        }
        logger.info(f"Task {self.request.id} completed successfully")
        return result
    except Exception as e:
        logger.error(f"Task {self.request.id} failed: {str(e)}", exc_info=True)
        raise
