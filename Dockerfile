FROM python:3.10-slim

# Install system dependencies (git is required for some python packages)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast, reliable builds as root
RUN pip install --no-cache-dir uv

# Create a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Set environmental variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app

WORKDIR $HOME/app

# 1. Copy the full source code
COPY . $HOME/app

# 2. Install everything at once as ROOT
RUN uv pip install --no-cache-dir --system .

# 3. Set correct ownership for the unprivileged user
RUN chown -R user:user $HOME/app

# 4. Switch to the unprivileged "user"
USER user

# The validator looks for 'server' script entry point in pyproject.toml
CMD ["server"]