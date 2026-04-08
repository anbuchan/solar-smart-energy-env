FROM python:3.10-slim

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install uv for fast, reliable builds
RUN pip install --no-cache-dir uv

# Copy the dependency files first for caching
COPY --chown=user pyproject.toml uv.lock $HOME/app/

# Install the project and its dependencies (including openenv-core)
RUN uv pip install --no-cache-dir -e .

# Copy the rest of the app
COPY --chown=user . $HOME/app

# The validator looks for 'server' script entry point in pyproject.toml
# CMD ["server"] will run 'server/app.py:main'
CMD ["server"]