# Multi-stage build for aipack
FROM python:3.11-slim as builder

# Install uv
RUN pip install uv

# Set work directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-install-project

# Production stage
FROM python:3.11-slim

# Install runtime dependencies only
COPY --from=builder /app/.venv /app/.venv

# Set work directory
WORKDIR /app

# Copy project files
COPY . .

# Set Python path
ENV PATH="/app/.venv/bin:$PATH"

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Default command
CMD ["./entrypoint.sh"]