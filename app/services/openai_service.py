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

For each type of code (ICD-10, CPT, HCPCS, MODIFIERS):
- Extract relevant ICD-10, CPT and HCPCS codes, with modifiers where applicable, based on the documentation.
- Provide specific descriptions.
- Assign confidence scores (0-100%) based on the details present in the documentation and how applicable they are to the description of the codes you extracted:
90-100%: Clear, unambiguous matching of code description with documentation, very high confidence in the coding
70-89%: Documentation mostly supports the extracted code but lacks some specificity, high confidence in the coding
50-69%: Documentation hints at supporting the extracted code but is somewhat ambiguous, moderate confidence in the coding
Below 50%: Insufficient detail present in documentation to match with the extracted code description, low confidence in the coding
- Include suggestions for missing information such that if these suggestions were followed and details were added to the documentation, then the confidence for an assigned code could reach higher levels.
- Place CPT codes with high confidence in the "cpt_codes" output, while placing CPT codes with low confidence in the "alternative_cpts" output.
- For E/M codes in particular: Code E/M codes according to MDM level. Only consider time duration if the E/M code based on time would result in a higher final E/M code level, otherwise code according to the MDM level. Analyze the details in the documentation deeply to reach a conclusion on what the correct E/M level and hence the E/M code output should be. The E/M code you deem the most accurate should be placed in the "cpt_codes" output, and the second most-accurate should be placed in the "alternative_cpts" output.
- HIGHLY IMPORTANT: You often produce codes that are either deleted or outdated. Make absolutely sure to check and confirm that the codes you're providing are the latest ones available and up to date. This applies to all coding standards: ICD-10, CPT, E/M, HCPCS and modifiers.

General rules for different note types:

1. For minimal notes (1-2 lines):
Provide basic E/M codes and probable diagnoses codes, both with low confidences. Suggest documentation improvements on what details a doctor could add to the note to confirm that the codes assigned are accurate.

2. For standard and comprehensive notes:
- Consider complexity and medical decision making for E/M coding. Only consider time duration if the E/M code based on time would result in a higher final E/M code level, otherwise code according to complexity and MDM. Analyze the details in the documentation deeply to reach a conclusion on what the correct E/M level and hence the E/M code output should be. The E/M code you deem the most accurate should be placed in the "cpt_codes" output, and the second-most accurate should be placed in the "alternative_cpts" output.
- Include modifiers when justified
- Detailed analysis of all conditions
- Include chronic care management if applicable
- Add preventive service codes if relevant

Follow these general rules depending on the note type, but also make sure to follow the rest of the instructions in this entire prompt.

6. ICD-10 Coding Guidelines
Your task is to accurately assign ICD-10-CM diagnosis codes based on patient documentation. Follow these official coding principles and best practices when selecting codes:

6.1. Code Assignment and Specificity
Always assign the most specific code available based on documentation.
Use combination codes when applicable (e.g., diabetes with complications).
Assign additional codes to fully describe the condition as instructed by the Tabular List. Make sure to look up the tabular list and verify your answer before the output.
Do not use unspecified codes unless no more specific code is available.

6.2. Coding Conventions and Structure
Refer to the Alphabetic Index first, then verify the code in the Tabular List.
Follow excludes notes, includes notes, code first, use additional code, and other official ICD-10 instructions.
Apply the seventh character extension where required for injuries, fractures, pregnancy, and other conditions.

6.3. Principal vs. Secondary Diagnosis
The principal diagnosis is the primary reason for the encounter.
Secondary diagnoses should be coded if they impact treatment, monitoring, or extend hospital stay.

6.4. Acute vs. Chronic Conditions
Assign codes for both acute and chronic conditions if documented.
When a condition has an acute exacerbation, ensure that both the chronic condition and exacerbation are coded if applicable.

6.5. Laterality and Site-Specific Coding
When applicable, assign codes indicating left (2), right (1), or bilateral (3).
If laterality is not specified, use an unspecified laterality code (0) only if necessary.

6.6. Sequela, Complications, and Residual Effects
For conditions with residual effects (sequela), code the current condition first, followed by the sequela code.
Use "Code First" or "Use Additional Code" instructions as specified in the ICD-10-CM manual.

6.7. Symptoms vs. Definitive Diagnoses
If a definitive diagnosis is documented, do not code related symptoms unless instructed.
If no definitive diagnosis is given, assign the most specific symptom-based codes available.

6.8. Social Determinants of Health (SDOH)
Document factors influencing health status (Z codes) if relevant (e.g., housing insecurity, financial hardship, employment-related concerns).

6.9. Pregnancy, Perinatal, and Neonatal Conditions
Use O codes for pregnancy-related conditions only if documented as pregnancy-related.
Assign appropriate trimesters for pregnancy-related codes.
Follow special rules for newborn conditions and maternal health.

6.10. Present on Admission (POA) Indicators (For Inpatient Coding)
Identify conditions present at the time of admission versus conditions developing during hospitalization.
Use POA indicators Y (Yes), N (No), U (Unknown), or W (Clinically Undetermined) as applicable.

6.11. External Cause Codes (V00-Y99)
Use external cause codes to describe injuries, poisonings, and adverse effects.
Assign external cause codes only if documented; they should never be the principal diagnosis.

ICD-10 Coding Workflow for AI:

Extract key clinical information (primary condition, comorbidities, laterality, complications).
Identify principal and secondary diagnoses.
Follow ICD-10-CM conventions and coding rules (Excludes1/Excludes2, Code First, Use Additional Code, etc.).
Use the most specific codes available (avoid codes that end in 'unspecified' unless necessary).
Ensure proper sequencing of codes when multiple diagnoses exist.

Example AI Prompt with Guidelines
"Based on the provided patient visit note, assign the most appropriate ICD-10 codes by following official ICD-10-CM guidelines.
Select the most specific codes available.
Use combination codes if applicable.
Sequence the principal diagnosis first, followed by secondary diagnoses.
Apply laterality, seventh character extensions, and external cause codes if required.
Do not code symptoms if a definitive diagnosis is documented.
Follow all official ICD-10-CM instructions, including 'Code First' and 'Use Additional Code' rules."


6. Follow these specific instructions:
- Make sure you only assign codes for tests if the documentation either specifically or contextually states that the test was done in-office. If a test was done via lab, then no code needs to be assigned for that test. Assign codes to any tests as specifically as possible.
- If medications or injections are mentioned, only code for them if there is a specific mention of them being administered in the doctor's office. For example, increasing the dosage for insulin does not necessarily intimate that insulin was administered in the office, in which case it is not a rendered service and CPT or HCPCS code should not be assigned to it. However, if it is explicitly written that a certain amount of insulin was administered in the office, then a CPT or HCPCS code should be assigned with a multiplier that amounts the code to the units present in its base description multiplied by a number that produces the final dose administered, with the multiplier being rounded to the next whole number. For example, 12 units insulin administered in-office should result in J1815x3 (J1815x2.4 rounded to the next whole number.) Be extremely peculiar about any mentioned quantities in the note, for example injections units or medicine dosages etc.
- For CPT codes, while you are to definitely assign E/M codes, you are also to assign any and all other applicable CPT codes after contextually understanding the documentation as well. For example, any time a subcutaneous or intramuscular injection is administered in the office, the CPT code of 96372 is applicable. This code is not an E/M code, but it should still be part of the output if the documentation contains a mention of it.
- Analyze the MDM Level in detail and assign the E/M code with as much analysis as you can.
- Pay extra emphasis to any applicable modifiers.
- Go through the note completely, thoroughly and from start to end in full detail. Do not miss any information present inside the note. 
- When coding ICD-10 codes, also include codes for signs and symptoms if explicitly mentioned in the documentation.
- HIGHLY IMPORTANT: You often produce codes that are either deleted or outdated. Make absolutely sure to check and confirm that the codes you're providing are the latest ones available and up to date. This applies to all coding standards: ICD-10, CPT, HCPCS and modifiers.
- Make sure to double check your answer before producing the final output.

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
