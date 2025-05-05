# Dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your repo
COPY . .

# Tell Streamlit to run your dashboard
ENV STREAMLIT_SERVER_HEADLESS=true
ENTRYPOINT ["streamlit", "run", "dashboard/app.py"]
