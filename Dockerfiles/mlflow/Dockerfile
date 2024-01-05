FROM ghcr.io/mlflow/mlflow:latest

RUN pip install psycopg2-binary

CMD -- mlflow server \
    --backend-store-uri postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@postgres:5432/mlflow \
    --default-artifact-root /mlflow/ \
    --host 0.0.0.0
