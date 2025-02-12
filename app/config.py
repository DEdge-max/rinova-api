from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # OpenAI Settings
    openai_api_key: str
    model_name: str = "gpt-4-turbo-preview"
    environment: str = "development"
    
    # MongoDB Settings
    mongodb_url: str
    mongodb_db_name: str = "rinova"
    
    # API Settings
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Logging Settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Performance Settings
    max_concurrent_extractions: int = 10
    extraction_timeout: int = 30  # seconds
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings():
    return Settings()

# Logging configuration function
def setup_logging(settings: Settings = get_settings()):
    import logging
    
    logging_level = getattr(logging, settings.log_level.upper())
    logging.basicConfig(
        level=logging_level,
        format=settings.log_format
    )
    return logging.getLogger(settings.mongodb_db_name)