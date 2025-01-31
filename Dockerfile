# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

RUN apt-get update --fix-missing && apt-get install -y --fix-missing build-essential
RUN apt-get -y install git

# Install necessary Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port used by FastAPI
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
