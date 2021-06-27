FROM python:3.8-slim

# Create a new user
RUN useradd --create-home --shell /bin/bash app_user

# use /home/app_user as the root directory for the project in the container.
WORKDIR /home/app_user

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# Change to newly created user
USER app_user

# Copy application source code from Dockerfile directory in the host to the /home/app_user directory in the container
COPY . .

# Set bash as default command, which will be invoked when docker container runs.
CMD ["bash"]

