import os
from urllib.parse import quote_plus

from celery import Celery
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD") or os.getenv("MYSQL_ROOT_PASSWORD", "")
MYSQL_HOST = os.getenv("MYSQL_HOST", "0.0.0.0")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "demo_bot")
CLOUD_SQL_CONNECTION_NAME = os.getenv("CLOUD_SQL_CONNECTION_NAME", "").strip()

# Celery settings
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379")

def build_database_url():
    configured_url = os.getenv("SQLALCHEMY_DATABASE_URL", "").strip()
    if configured_url:
        return configured_url

    user = quote_plus(MYSQL_USER)
    password = quote_plus(MYSQL_PASSWORD)
    database = quote_plus(MYSQL_DATABASE)

    if CLOUD_SQL_CONNECTION_NAME:
        socket_path = quote_plus(f"/cloudsql/{CLOUD_SQL_CONNECTION_NAME}")
        return f"mysql+pymysql://{user}:{password}@/{database}?unix_socket={socket_path}"

    return f"mysql+pymysql://{user}:{password}@{MYSQL_HOST}:{MYSQL_PORT}/{database}"


# MySQL database configuration
SQLALCHEMY_DATABASE_URL = build_database_url()

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, pool_pre_ping=True  # Improve connection resilience
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_celery_app(name):
    # Create a Celery app instance
    app = Celery(
        name,
        broker=CELERY_BROKER_URL,  # Redis as the message broker
        backend=CELERY_RESULT_BACKEND,  # Redis as the result backend
    )

    # Optionally, you can configure additional settings here
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="Asia/Ho_Chi_Minh",  # Set to a city in UTC+7
        enable_utc=True,
    )

    # Configure Celery logging
    app.conf.update(
        worker_hijack_root_logger=False,
        worker_log_format="[%(asctime)s: %(levelname)s/%(processName)s] %(message)s",
        worker_task_log_format="[%(asctime)s: %(levelname)s/%(processName)s] %(message)s",
    )

    return app
