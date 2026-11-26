# Use a small Python base image.
FROM python:3.12-slim

# Create and use /app as the working directory inside the image.
WORKDIR /app

ENV PYTHONPATH=/app

# Install dependencies before copying the application.
# This allows Docker to reuse the dependency layer when only code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code, configs, data, and saved model
COPY configs/ ./configs/
COPY src/ ./src/
COPY tests/ ./tests/
COPY data/ ./data/
COPY models/ ./models/

# Train the model into the image
RUN python -m src.train

# Default command runs the automated test suite
CMD ["pytest", "-v"]