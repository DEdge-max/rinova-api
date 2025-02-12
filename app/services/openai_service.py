from openai import AsyncOpenAI
from typing import Dict
import json
from ..config import get_settings

settings = get_settings()
client = AsyncOpenAI(api_key=settings.openai_api_key)

class OpenAIService:
    def __init__(self):
        self.client = client
        self.model = settings.model_name

    async def extract_medical_codes(self, medical_text: str) -> Dict:
        """Extract ICD-10 and CPT codes from medical text using OpenAI."""
        system_prompt = """
        You are a medical coding expert. Extract relevant ICD-10 and CPT codes from the given text.
        
        Guidelines:
        1. Start with what's explicitly stated in the text
           - Main symptoms or complaints
           - Any diagnosed conditions
           - Any procedures or tests mentioned
        
        2. Code Assignment Rules:
           - Only code what is documented
           - If chief complaint is clear, mark it as primary
           - For brief notes, it's okay to have just one code
           - Match E&M level to documentation detail
           - Include ordered tests/procedures when mentioned
        
        3. Confidence Scoring:
           - High (>0.9): Clear documentation with specific details
           - Medium (0.7-0.9): Some supporting information
           - Low (<0.7): Minimal information or unclear context
        
        Return in this format:
        {
            "icd10_codes": [
                {
                    "code": "[code]",
                    "description": "[official description]",
                    "confidence": [0-1],
                    "primary": [true/false],
                    "evidence": "[relevant text from note]"
                }
            ],
            "cpt_codes": [
                {
                    "code": "[code]",
                    "description": "[official description]",
                    "confidence": [0-1],
                    "category": "[category]",
                    "evidence": "[relevant text from note]"
                }
            ]
        }
        """

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Extract codes from this text: {medical_text}"}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return self._validate_extraction_result(result)
            
        except Exception as e:
            raise Exception(f"Error in code extraction: {str(e)}")

    def _validate_extraction_result(self, result: Dict) -> Dict:
        """
        Validate and clean up the extraction results.
        """
        if not isinstance(result, dict):
            raise ValueError("Invalid extraction result format")

        # Ensure required keys exist
        required_keys = ['icd10_codes', 'cpt_codes']
        for key in required_keys:
            if key not in result:
                result[key] = []

        # Validate confidence scores
        for code_list in result.values():
            for code in code_list:
                if 'confidence' in code:
                    code['confidence'] = max(0.0, min(1.0, float(code['confidence'])))

        return result