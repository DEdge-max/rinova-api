from openai import AsyncOpenAI
from typing import Dict
import json
from ..config import get_settings, setup_logging
import logging

logger = setup_logging()

class OpenAIService:
    def __init__(self):
        settings = get_settings()
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.model_name
        self.timeout = settings.extraction_timeout
        
    async def extract_medical_codes(self, medical_text: str) -> Dict:
        """
        Extract ICD-10 and CPT codes from medical text using OpenAI.
        Includes timeout and error handling based on config settings.
        """
        system_prompt = """
        You are a medical coding expert. Extract relevant ICD-10 and CPT codes from the given text.
        Return the codes in JSON format.
        
        Required format for response:
        {
            "icd10_codes": [
                {
                    "code": "code",
                    "description": "description",
                    "confidence": 0.95,
                    "primary": true
                }
            ],
            "cpt_codes": [
                {
                    "code": "code",
                    "description": "description",
                    "confidence": 0.9,
                    "category": "category"
                }
            ]
        }
        
        Guidelines:
        1. Always return valid JSON with the word 'json' in the explanation
        2. Include only documented conditions and procedures
        3. Set confidence scores based on documentation clarity
        4. Mark primary diagnoses where clear
        """

        try:
            logger.info(f"Starting code extraction for text of length: {len(medical_text)}")
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Please analyze this medical text and return JSON containing the extracted codes: {medical_text}"}
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
                timeout=self.timeout
            )
            
            result = json.loads(response.choices[0].message.content)
            validated_result = self._validate_extraction_result(result)
            
            logger.info(
                f"Successfully extracted {len(validated_result.get('icd10_codes', []))} ICD-10 codes and "
                f"{len(validated_result.get('cpt_codes', []))} CPT codes"
            )
            
            return validated_result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI response: {e}")
            raise Exception(f"Invalid JSON response from code extraction: {e}")
        except TimeoutError:
            logger.error(f"Extraction timed out after {self.timeout} seconds")
            raise Exception(f"Code extraction timed out after {self.timeout} seconds")
        except Exception as e:
            logger.error(f"Error in code extraction: {str(e)}")
            raise Exception(f"Error in code extraction: {str(e)}")

    def _validate_extraction_result(self, result: Dict) -> Dict:
        """
        Validate and clean up the extraction results.
        Ensures all required fields are present and properly formatted.
        """
        if not isinstance(result, dict):
            raise ValueError("Invalid extraction result format")

        # Initialize with empty lists if keys missing
        validated_result = {
            'icd10_codes': [],
            'cpt_codes': []
        }

        # Validate ICD-10 codes
        if 'icd10_codes' in result and isinstance(result['icd10_codes'], list):
            for code in result['icd10_codes']:
                try:
                    validated_code = {
                        'code': str(code.get('code', '')).strip(),
                        'description': str(code.get('description', '')).strip(),
                        'confidence': max(0.0, min(1.0, float(code.get('confidence', 0.5)))),
                        'primary': bool(code.get('primary', False))
                    }
                    if validated_code['code']:  # Only add if code is not empty
                        validated_result['icd10_codes'].append(validated_code)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping invalid ICD-10 code entry: {e}")
                    continue

        # Validate CPT codes
        if 'cpt_codes' in result and isinstance(result['cpt_codes'], list):
            for code in result['cpt_codes']:
                try:
                    validated_code = {
                        'code': str(code.get('code', '')).strip(),
                        'description': str(code.get('description', '')).strip(),
                        'confidence': max(0.0, min(1.0, float(code.get('confidence', 0.5)))),
                        'category': str(code.get('category', 'Unspecified')).strip()
                    }
                    if validated_code['code']:  # Only add if code is not empty
                        validated_result['cpt_codes'].append(validated_code)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping invalid CPT code entry: {e}")
                    continue

        return validated_result
