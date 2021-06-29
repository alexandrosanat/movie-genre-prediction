FROM python:3.8-slim

# Create a new user
RUN useradd --create-home --shell /bin/bash app_user

# Use /home/app_user as the root directory for the project in the container.
WORKDIR /home/app_user

COPY requirements.txt ./
# Copy all data required by the nltk library
COPY ./nltk_data /app_user/AppData/Roaming/nltk_data

RUN pip install --no-cache-dir -r requirements.txt

# Change to newly created user
USER app_user

# Copy source code from Dockerfile directory to the /home/app_user directory in the container
COPY . .

RUN python -c "import nltk; nltk.download('punkt')"

# Set bash as default command, which will be invoked when docker container runs.
CMD ["bash"]

