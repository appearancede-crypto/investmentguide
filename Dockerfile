# Signal Engine — hosted web app.
# Builds a small image that serves the live Signal Engine page + deep-dive API.
FROM python:3.13-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PAGE_CACHE_TTL=45

# Install only the web server's dependencies (no Streamlit/Plotly/matplotlib).
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

# App code + config. The candle DB is built at runtime (see CMD), not baked in.
COPY crypto_tool ./crypto_tool
COPY config.yaml ./config.yaml
RUN mkdir -p data

# Documentation only — the host injects the real port via $PORT, which the
# server reads automatically. Binds 0.0.0.0.
EXPOSE 8787

# On boot: ingest candles if the DB is empty (1500/coin keeps cold starts quick),
# then serve and refresh the data every 20 minutes.
CMD ["python", "-m", "crypto_tool.cli", "web", "--serve", "--no-open", \
     "--ensure-data", "--ingest-limit", "1500", "--refresh-min", "20"]
