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

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user requirements.txt $HOME/app/

# Install dependencies
RUN pip install --no-cache-dir numpy==1.26.4
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app with correct permissions
COPY --chown=user . $HOME/app

CMD ["python", "app.py"]