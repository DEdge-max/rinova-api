import asyncio
import json
import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.openai_service import OpenAIService

async def test_extraction():
    service = OpenAIService()
    test_text = "Patient presented with type 2 diabetes mellitus. Performed routine follow-up office visit."
    
    try:
        result = await service.extract_medical_codes(test_text)
        print("Extraction result:", json.dumps(result, indent=2))
    except Exception as e:
        print("Error:", str(e))

# Run the test
if __name__ == "__main__":
    asyncio.run(test_extraction())