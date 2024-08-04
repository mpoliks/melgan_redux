FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN sed -i '/argparse/d' requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Copy the rest of the application
COPY . .

# Set default command
CMD ["python", "run.py"]

