# 1. Start from an official Python base image
FROM python:3.8
#FROM --platform=linux/amd64 python:3.8-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your application code into the container
COPY . .

# 5. Expose the port the app runs on
EXPOSE 8000

# 6. Define the command to run your app when the container starts
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]