FROM python:3.9-slim

# Set working directory
WORKDIR /code

# Install build dependencies
RUN apt-get update && apt-get install -y gcc

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the application code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
