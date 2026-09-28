FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY riverforge ./riverforge
COPY tests ./tests
COPY pyproject.toml README.md ./

CMD ["python", "-m", "riverforge.examples.run_demo"]
