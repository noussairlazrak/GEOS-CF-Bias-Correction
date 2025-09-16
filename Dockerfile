FROM ghcr.io/astral-sh/uv:debian-slim

ADD . /app

WORKDIR /app

RUN apt-get -y update \
      && apt-get install -y --no-install-recommends \
      libgomp1 \
      && rm -rf /var/lib/apt/lists/*

RUN uv sync --locked

CMD ["uv", "run", "python3"]
