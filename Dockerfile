# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy requirements file
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 🔥 Manually download and install `punkt`
RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt_tab

# Ensure the path is set correctly
ENV NLTK_DATA="/usr/local/share/nltk_data"

# Copy application code
COPY . .

# Expose the app port
EXPOSE 5000

# Run the application
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
