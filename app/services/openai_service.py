import logging
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

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class OpenAIService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = "gpt-4-turbo-preview"

    async def extract_codes(self, note_text: str) -> CodeExtractionResult:
        """
        Extract medical codes from the provided note text using OpenAI's GPT-4.

        Args:
            note_text (str): The medical note text to analyze

        Returns:
            CodeExtractionResult: Extracted medical codes with confidence scores
        """
        try:
            logger.info("Starting code extraction for medical note")

            system_prompt = """You are a highly specialized AI assistant designed to analyze medical documentation and extract structured coding information in JSON format. Your primary task is to process clinical notes written in various formats (e.g., SOAP, HPI, CC) and generate the following outputs with precision and clarity. Your output must always be generated, regardless of the length, completeness, or quality of the note. Never return an error.

For each type of code (ICD-10, CPT, HCPCS):
- Extract all relevant codes based on the documentation
- Provide specific descriptions
- Assign confidence scores (0-100%) based on documentation quality:
  * 90-100%: Clear, unambiguous documentation
  * 70-89%: Supports but lacks some specificity
  * 50-69%: Ambiguous, moderate confidence
  * Below 50%: Insufficient detail
- Include suggestions for missing information
- For CPT codes, provide alternative codes with justification

Rules for different note types:
1. For minimal notes (1-2 lines):
   - Provide basic E/M codes with low confidence scores
   - Suggest documentation improvements
   - Include probable diagnoses with very low confidence

2. For standard notes:
   - Extract all explicit diagnoses and procedures
   - Consider complexity and time for E/M coding
   - Include modifiers when justified

3. For comprehensive notes:
   - Detailed analysis of all conditions
   - Consider medical decision making
   - Include chronic care management if applicable
   - Add preventive service codes if relevant

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

            user_prompt = f"Analyze this medical note and extract all relevant medical codes. Always generate codes even if the note is very brief:\n\n{note_text}"

            logger.info("Calling OpenAI API for code extraction...")
            response = await self._get_completion(system_prompt, user_prompt)
            logger.info("Received response from OpenAI")
            logger.debug(f"Raw OpenAI response: {response}")

            # Parse the JSON response
            try:
                extraction_data = json.loads(response)
                logger.info("Successfully parsed JSON response")
                logger.debug(f"Parsed extraction data: {json.dumps(extraction_data, indent=2)}")

                result = CodeExtractionResult(
                    icd10_codes=[ICD10Code(**code) for code in extraction_data.get("icd10_codes", [])],
                    cpt_codes=[CPTCode(**code) for code in extraction_data.get("cpt_codes", [])],
                    alternative_cpts=[AlternativeCPT(**code) for code in extraction_data.get("alternative_cpts", [])],
                    modifiers=[Modifier(**mod) for mod in extraction_data.get("modifiers", [])],
                    hcpcs_codes=[HCPCSCode(**code) for code in extraction_data.get("hcpcs_codes", [])]
                )

                logger.info(f"Extracted {len(result.icd10_codes)} ICD-10 codes, "
                          f"{len(result.cpt_codes)} CPT codes, "
                          f"{len(result.hcpcs_codes)} HCPCS codes, "
                          f"{len(result.modifiers)} modifiers, and "
                          f"{len(result.alternative_cpts)} alternative CPT codes.")

                return result

            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                logger.error(f"Raw response that failed parsing: {response}")
                raise ValueError("Failed to parse OpenAI response as JSON")

            except Exception as e:
                logger.error(f"Error processing response: {str(e)}")
                logger.error(f"Extraction data that caused error: {json.dumps(extraction_data, indent=2) if 'extraction_data' in locals() else 'Not available'}")
                raise ValueError(f"Failed to process OpenAI response: {str(e)}")

        except Exception as e:
            logger.error(f"Error in extract_codes: {str(e)}")
            raise

    async def _get_completion(self, system_prompt: str, user_prompt: str) -> str:
        """
        Get completion from OpenAI API with error handling and logging.
        """
        try:
            logger.info(f"Requesting completion from OpenAI with model: {self.model}")
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=4000,
                response_format={"type": "json_object"}
            )
            logger.info("Successfully received completion from OpenAI")
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise ValueError(f"Failed to get OpenAI completion: {str(e)}")

# Create a singleton instance
openai_service = OpenAIService()
