FROM python:3.11-slim

# Create a non-root user with UID 1000 (standard for Hugging Face Spaces & security compliance)
RUN useradd -m -u 1000 user

# Switch to non-root user
USER user

# Configure environment variables for universal cloud compatibility
ENV PATH="/home/user/.local/bin:$PATH" \
    MCP_TRANSPORT="sse" \
    PORT=8080 \
    HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /home/user/app

# Copy requirements and install dependencies
COPY --chown=user requirements.txt /home/user/app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-cache SentenceTransformer embedding weights inside image for instant, zero-latency startup
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application files
COPY --chown=user . /home/user/app

# Expose standard cloud ports (8080: CreateOS/GCP/AWS/mcphosting, 7860: Hugging Face Spaces, 3000: Node/General)
EXPOSE 8080 7860 3000

# Start the FastMCP server with automatic transport and port resolution
CMD ["python", "src/server.py"]

