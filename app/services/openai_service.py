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
        self.model = "gpt-4o-2024-08-06"

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

CRITICAL VERIFICATION REQUIREMENT: You often produce outdated or deleted codes. ALWAYS confirm each code is current and valid against the latest standards before finalizing your response.

For each type of code (ICD-10, CPT, HCPCS, MODIFIERS):
- Extract relevant codes based on documented evidence
- Provide specific descriptions
- Assign confidence scores (0-100%) based on documentation strength:
  * 90-100%: Clear, unambiguous matching with documentation
  * 70-89%: Documentation mostly supports but lacks some specificity
  * 50-69%: Documentation hints at supporting but somewhat ambiguous
  * Below 50%: Insufficient detail to match with code description
- Include suggestions for missing information to improve confidence levels
- For CPT codes, provide alternative codes with justification

E/M CODE SELECTION RULES:
- Determine if NEW (99201-99205) or ESTABLISHED (99211-99215) patient
- Assess BOTH MDM complexity AND time spent
- For MDM complexity, evaluate using 2/3 element rule:
  * Problem complexity (minor, stable chronic, acute, exacerbation, life-threatening)
  * Data complexity (records review, test ordering/review, consultation)
  * Risk level (minimal, low, moderate, high)
- ALWAYS select the HIGHER code between time-based and complexity-based options

NEW PATIENT IDENTIFICATION:
- Look for explicit mentions of "new patient," "initial visit," or "consultation" 
- When a patient is identified as new, use 99201-99205 series codes ONLY

HIGH COMPLEXITY INDICATORS FOR NEW PATIENTS (99205):
- End-stage organ failure (e.g., ESRD, end-stage heart failure)
- Poorly controlled diabetes (A1C >8.0%)
- Multiple chronic conditions requiring medication management
- When a new patient presents with BOTH diabetes AND end-stage renal disease, this ALWAYS qualifies for high complexity (99205)

ICD-10 CODING REQUIREMENTS:
- Use most specific codes available (avoid "unspecified" when possible)
- Apply combination codes when conditions are related
- Follow proper sequencing (primary reason for encounter first)
- Include secondary conditions that impact treatment
- Apply 7th character extensions and laterality indicators when required

CPT/HCPCS SPECIFIC RULES:
- Only code labs if documentation states blood was drawn AND tested DURING visit
- Code injections (96372) when administered in-office
- Apply multipliers for medication units (e.g., J1815x2 for 10 units insulin)
- Include all in-office procedures (not just E/M codes)

VERIFICATION CHECKLIST:
1. All codes are current and valid (not deleted/outdated)
2. New vs. established patient status correctly identified
3. Highest appropriate E/M code selected between time and complexity
4. All in-office procedures coded
5. Combination codes used when applicable
6. ICD-10 codes properly sequenced

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
            response = self.client.chat.completions.create(
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
