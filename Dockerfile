FROM python:3.10-slim

COPY shoppingassistant/ /shoppingassistant/
COPY requirements.txt /requirements.txt
COPY api/fast.py /fast.py
COPY models/ /models/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["uvicorn", "fast:app", "--host", "0.0.0.0", "--port", "8000"]
