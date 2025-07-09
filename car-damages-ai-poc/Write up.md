#### Python AI Microservice (FastAPI)

- **Framework:** FastAPI
    
- **Libraries:**
    
    - Hugging Face Transformers (ViT)
        
    - Torch (PyTorch for model inference)
        
    - PIL or OpenCV (image processing)
        
    - NumPy (vector ops)
        
    - Uvicorn (ASGI server)
        
    - Pydantic (input validation)
        
- **Features:**
    
    - `/predict-damage` endpoint returns binary classification with confidence
        
    - `/compare` endpoint returns similarity score between two photos
        
    - Embedding caching & performance tuning
        
- **Deployment:**
    
    - Docker container (locally or on Google Cloud Run)
        
    - Logging via Google Cloud Logging or Sentry (optional)
        

#### Integration Stack

- **Cloud Platform:** Google Cloud (preferred)
    
- **Messaging (optional):** Pub/Sub or RabbitMQ for async processing
    
- **Monitoring:** Prometheus + Grafana, or Google Cloud Monitoring