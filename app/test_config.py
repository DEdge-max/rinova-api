from config import get_settings

settings = get_settings()
print(f"Environment: {settings.environment}")
print(f"Model name: {settings.model_name}")
print("API key exists:", bool(settings.openai_api_key))