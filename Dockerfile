FROM python:3.10

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
# Switch to the "user" user
USER user
# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the requirements file into the container
COPY --chown=user requirements.txt $HOME/app/

# Install dependencies from requirements.txt (where versions are pinned)
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app with correct permissions
COPY --chown=user . $HOME/app

# Start the application
CMD ["python", "app.py"]