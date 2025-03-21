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

Go through the note completely, thoroughly and from start to end in full detail. Do not miss any information present inside the note.
For each type of code (ICD-10, CPT, HCPCS, Modifiers):
Extract relevant ICD-10, CPT and HCPCS codes with modifiers where applicable based on the documentation.
Provide specific descriptions.
Assign confidence scores (0-100%) based on the details present in the documentation and how applicable they are to the description of the codes you extracted:
- 90-100%: Clear, unambiguous matching of code description with documentation, very high confidence in the coding
- 70-89%: Documentation mostly supports the extracted code but lacks some specificity, high confidence in the coding
- 50-69%: Documentation hints at supporting the extracted code but is somewhat ambiguous, medium confidence in the coding
- Below 50%: Insufficient detail present in documentation to match with the extracted code description, low confidence in the coding
Include suggestions for missing information such that if these suggestions were followed and details were added to the documentation, then the confidence for an assigned code could reach higher levels.
Place CPT codes with high confidence in the "cpt_codes" output, while placing CPT codes with low confidence in the "alternative_cpts" output.
For E/M codes in particular: If an E/M code is applicable to the note, then analyze the complexity of the note in detail and provide the E/M code according to the MDM level. For additional context regarding outpatient office visits: remember that 992X2 is straightforward MDM, 992X3 is low MDM, 992X4 is moderate MDM, and 992X5 is high MDM. Determine the MDM complexity according to the latest E/M guidelines and assign the relevant code when applicable.
Keep the following in context:
1. Only code for details explicitly mentioned in the note. Do not code based on if something was done.
2. If medications, topicals or injections are mentioned, only code for them if the note mentions them being provided, applied or administered. If they're only advised, prescribed, changed or have their dosage adjusted, then they do not need their own codes, but should still be considered for MDM complexity calculations. For example, increasing the dose of insulin does not mean insulin was administered in the office, and a code for insulin should only be added if there is an explicit mention of insulin being administered in the note. The dose adjustment should still be taken into account for MDM complexity determination, however.
3. If a note mentions a test, only code for it if there is an explicit mention of the test being done in the provider's office. If it's contextually understood that the advised test is meant to be carried out at a lab or other facility, then there's no need to code for it. However, it should still be considered for purposes of MDM complexity determination.
4. Take any quantities mentioned in the note into consideration. If it is explicitly written that a certain amount of an injection was administered in the office, then a HCPCS code should be assigned with a multiplier that amounts the code to the units present in its base description multiplied by a number that produces the final dose administered, with the multiplier being rounded to the next whole number. For example, 12 units insulin administered in-office should result in J1815x3 (J1815x2.4 having the multiplier (2.4) rounded to the next whole number (3).)
HIGHLY IMPORTANT: Your database of codes and guidelines is outdated. Make sure to search the internet to follow the latest available guidelines and to assign the latest available codes.
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
