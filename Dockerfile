# 1. Use Python 3.11 (Compatible with your newer dependencies)
FROM python:3.11-slim

# 2. Set the working directory
WORKDIR /app

# 3. Install system dependencies (libgl1 is correct for this version)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements
COPY requirements.txt .

# 5. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy application code
COPY . .

# 7. Expose port
EXPOSE 8501

# 8. Run the app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]