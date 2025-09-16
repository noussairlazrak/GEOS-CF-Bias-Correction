# GEOS-CF ML downscaled forecasts

## Development

1. Install `uv` if not installed.

2. Locally, sync and lock the python environment.

    ```sh
    uv sync
    uv lock
    ```

3. Confirm that everything works locally

    ```sh
    uv run python -c "from MLpred import mlpred, funcs"
    ```

4. Build Docker image:

    ```sh
    docker build -t mlpred:latest .
    ```

5. Test Docker image:

    ```sh
    docker run -it --rm mlpred:latest \
        uv run python3 -c \
        "from MLpred import mlpred, funcs"
    ```

6. Configure AWS:

    ```sh
    # Also set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_SESSION_TOKEN
    export AWS_REGION="us-west-2"
    export ECR_REPO="466270585360.dkr.ecr.us-west-2.amazonaws.com/geos-cf/mlpred"

    aws ecr get-login-password --region $AWS_REGION | \
        docker login --username AWS --password-stdin $ECR_REPO
    ```

7. Tag and push Docker images
    
    ```sh
    docker image tag mlpred:latest "$ECR_REPO:latest"
    docker push "$ECR_REPO:latest"
    ```
