# Use a lightweight Python base
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install the required libraries directly
RUN pip install --no-cache-dir openenv-core pydantic openai numpy fastapi uvicorn

COPY . .

RUN useradd -m -u 1000 user
USER user

EXPOSE 7860

# Start the OpenEnv server on the correct port
CMD ["openenv", "app:app", "--host", "0.0.0.0", "--port", "7860"]
