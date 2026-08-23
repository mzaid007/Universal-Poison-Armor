FROM python:3.11-slim

# Create a non-root user with UID 1000 (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user

# Switch to non-root user
USER user

# Ensure user local bin is in PATH and default to SSE transport for cloud container
ENV PATH="/home/user/.local/bin:$PATH" \
    MCP_TRANSPORT="sse" \
    PORT=7860

# Set working directory
WORKDIR /home/user/app

# Copy requirements and install dependencies
COPY --chown=user requirements.txt /home/user/app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY --chown=user . /home/user/app

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Start the FastMCP server with SSE transport
CMD ["python", "skills/ai-poison-defense/src/server.py"]
