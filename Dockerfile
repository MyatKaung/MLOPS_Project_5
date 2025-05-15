# Use a lightweight Python image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the application and its dependencies
RUN pip install --no-cache-dir -e .

# Expose port 5000 for the application
EXPOSE 5000

# Run the application
CMD ["python", "application.py"]
