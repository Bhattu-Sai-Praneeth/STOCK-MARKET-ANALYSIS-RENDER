FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install pandas-ta separately
RUN pip install git+https://github.com/twopirllc/pandas-ta.git

# Expose port for Streamlit
EXPOSE 7860

# Run Streamlit app
CMD ["streamlit", "run", "main.py", "--server.port=7860", "--server.address=0.0.0.0"]
