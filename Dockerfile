# Use an official Python runtime as a parent image
FROM python:3.11.3

# Set the environment variable to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Copy the dependencies file
COPY requirements.txt .

# Copy the folder
COPY . .

# Install any needed dependencies specified in requirements.txt
RUN pip install -r requirements.txt

CMD [ "python3", "app.py" ]