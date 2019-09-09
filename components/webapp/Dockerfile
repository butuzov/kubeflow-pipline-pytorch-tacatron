FROM butuzov/kf-ml:latest

RUN python3 -m pip install --no-cache-dir \
    "Flask"

COPY src /app
WORKDIR  /app

ENTRYPOINT ["python3", "main.py"]
