import os

# No MLflow server exists in CI: fail registry calls fast instead of grinding
# through the client's default 7-retry exponential backoff per request.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
