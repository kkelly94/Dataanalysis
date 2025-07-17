FROM python:3.10

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Set the PORT environment variable for Cloud Run
ENV PORT 8080
EXPOSE $PORT

# Update Streamlit to run on the specified PORT
CMD ["sh", "-c", "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true"]