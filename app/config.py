from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
import logging

class Settings(BaseSettings):
    # The model_config replaces the Config class in Pydantic v2
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True
    )

    # OpenAI Settings
    OPENAI_API_KEY: str
    model_name: str = "gpt-4-turbo-preview"
    environment: str = "development"
    
    # MongoDB Settings
    MONGODB_URL: str
    MONGODB_DB_NAME: str = "rinova"  # Changed to match the case in your code
    
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

@lru_cache()
def get_settings():
    return Settings()

# Instantiate settings
settings = get_settings()

# Logging configuration function
def setup_logging(settings: Settings = get_settings()):
    logging_level = getattr(logging, settings.log_level.upper())
    logging.basicConfig(
        level=logging_level,
        format=settings.log_format
    )
    return logging.getLogger(settings.MONGODB_DB_NAME)

# Initialize logger
logger = setup_logging()
