FROM python:3.7-slim

RUN apt-get update \
    && apt-get install -y libsndfile1

RUN python3 -m pip install --no-cache-dir \
    "numpy>=1.13.3" \
    "librosa>=0.5.1" \
    "falcon>=1.2.0" \
    "inflect>=0.2.5" \
    "scipy>=1.0.0" \
    "Unidecode>=0.4.21" \
    "pandas>=0.21.0" \
    "SoundFile" \
    "torch>=0.4.1.post2"
