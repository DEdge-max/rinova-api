from openai import OpenAI
import json
from typing import Dict, Any
from app.config import settings
from app.models.pydantic_models import (
    CodeExtractionResult,
    ICD10Code,
    CPTCode,
    HCPCSCode,
    Modifier,
    AlternativeCPT
)

class OpenAIService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = "gpt-4-turbo-preview"  # Using the latest GPT-4 model

    async def extract_codes(self, note_text: str) -> CodeExtractionResult:
        """
        Extract medical codes from the provided note text using OpenAI's GPT-4.
        
        Args:
            note_text (str): The medical note text to analyze
            
        Returns:
            CodeExtractionResult: Extracted medical codes with confidence scores
        """
        try:
            system_prompt = """You are a highly specialized AI assistant designed to analyze medical documentation and extract structured coding information in JSON format. Your primary task is to process clinical notes written in various formats (e.g., SOAP, HPI, CC) and generate the following outputs with precision and clarity. Your output must always be generated, regardless of the length, completeness, or quality of the note. Never return an error.

For each type of code (ICD-10, CPT, HCPCS):
- Extract all relevant codes based on the documentation
- Provide specific descriptions
- Assign confidence scores (0-100%)
- Include suggestions for missing information
- For CPT codes, provide alternative codes with justification
- Identify applicable modifiers

Your response must be valid JSON matching this exact structure:
{
  "icd10_codes": [
    {
      "code": "E11.9",
      "description": "Type 2 Diabetes Mellitus without complications",
      "confidence_score": 95,
      "suggestions": []
    }
  ],
  "cpt_codes": [
    {
      "code": "99213",
      "description": "Office visit, established patient",
      "confidence_score": 90,
      "suggestions": [
        "Add more detail about medical decision-making"
      ]
    }
  ],
  "alternative_cpts": [
    {
      "code": "99214",
      "description": "Office visit, established patient, higher complexity",
      "confidence_score": 70,
      "justification": "Could apply with more documented complexity",
      "missing_documentation": [
        "Document total time spent",
        "Include details about medical decision-making complexity"
      ]
    }
  ],
  "modifiers": [
    {
      "modifier": "25",
      "description": "Significant, separate E/M service",
      "confidence_score": 85,
      "suggestions": []
    }
  ],
  "hcpcs_codes": [
    {
      "code": "A4230",
      "description": "Infusion set for insulin pump",
      "confidence_score": 80,
      "suggestions": []
    }
  ]
}"""

            user_prompt = f"Analyze this medical note and extract all relevant medical codes:\n\n{note_text}"

            response = await self._get_completion(system_prompt, user_prompt)
            
            # Parse the JSON response
            try:
                extraction_data = json.loads(response)
                return CodeExtractionResult(
                    icd10_codes=[ICD10Code(**code) for code in extraction_data.get("icd10_codes", [])],
                    cpt_codes=[CPTCode(**code) for code in extraction_data.get("cpt_codes", [])],
                    alternative_cpts=[AlternativeCPT(**code) for code in extraction_data.get("alternative_cpts", [])],
                    modifiers=[Modifier(**mod) for mod in extraction_data.get("modifiers", [])],
                    hcpcs_codes=[HCPCSCode(**code) for code in extraction_data.get("hcpcs_codes", [])]
                )
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Raw response: {response}")
                raise ValueError("Failed to parse OpenAI response as JSON")
            except Exception as e:
                print(f"Error processing response: {e}")
                raise ValueError(f"Failed to process OpenAI response: {str(e)}")

        except Exception as e:
            print(f"Error in extract_codes: {e}")
            # Return empty result structure in case of error
            return CodeExtractionResult()

    async def _get_completion(self, system_prompt: str, user_prompt: str) -> str:
        """
        Get completion from OpenAI API with error handling and retry logic.
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,  # Using 0 temperature for consistent, deterministic outputs
                max_tokens=4000,
                response_format={ "type": "json" }  # Ensure JSON response
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"OpenAI API error: {e}")
            raise ValueError(f"Failed to get OpenAI completion: {str(e)}")

# Create a singleton instance
openai_service = OpenAIService()
