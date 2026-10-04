FROM python:3.12-slim
WORKDIR /app
RUN useradd --create-home --uid 10001 appuser
COPY --chown=appuser:appuser pyproject.toml uv.lock README.md ./
RUN pip install --no-cache-dir uv==0.12.21 && uv sync --frozen --no-dev
COPY --chown=appuser:appuser src ./src
USER appuser
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["uvicorn","rcs.server.app:app","--host","0.0.0.0","--port","8000"]
