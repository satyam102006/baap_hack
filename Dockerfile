# Use a lightweight Python base
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install the required libraries directly
RUN pip install --no-cache-dir openenv-core pydantic openai numpy fastapi uvicorn

# Copy your entire repository into the container
COPY . .

# Hugging Face Spaces require a non-root user running on UID 1000
RUN useradd -m -u 1000 user
USER user

# Expose the specific port Hugging Face looks for
EXPOSE 7860

# Start the OpenEnv server on the correct port
CMD ["openenv", "serve", "--host", "0.0.0.0", "--port", "7860"]