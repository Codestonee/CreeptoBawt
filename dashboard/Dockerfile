FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY dashboard/frontend/package*.json ./
RUN npm ci
COPY dashboard/frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
COPY dashboard/backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY dashboard/backend/ ./dashboard/backend/
COPY --from=frontend-builder /app/frontend/dist ./static
EXPOSE 8000
CMD ["python", "-m", "dashboard.backend.main"]
