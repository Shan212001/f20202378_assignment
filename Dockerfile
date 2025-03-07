# TASK DONE BY SHASHANK GAUTAM (2020B5A32378H)

FROM python:3.9-slim

WORKDIR /app

COPY ./app /app

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
