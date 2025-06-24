FROM python:3.9-slim
WORKDIR /Fastapi
COPY requirementsgoat.txt .
RUN pip install --no-cache-dir -r requirementsgoat.txt
COPY . .
EXPOSE 80

CMD [ "uvicorn", "application:app", "--host", "0.0.0.0", "--port", "80" ] 