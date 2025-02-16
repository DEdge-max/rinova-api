from openai import AsyncOpenAI
from typing import Dict
import json
from ..config import get_settings, setup_logging
import logging

logger = setup_logging()

class OpenAIService:
    def __init__(self):
        settings = get_settings()
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.model_name
        self.timeout = settings.extraction_timeout
        
    async def extract_medical_codes(self, medical_text: str) -> Dict:
        """
        Extract ICD-10 and CPT codes from medical text using OpenAI.
        Includes timeout and error handling based on config settings.
        """
        system_prompt = """
        You are a medical coding expert. Analyze medical documentation to extract codes, providing clear rationale for each code and identifying documentation improvements needed. Return response as a valid JSON object only. Do not include any explanatory text before or after the JSON object. Do not use markdown formatting or code blocks.

        Required format for response:
        {
            "note_type": "brief|comprehensive|emergency|operative|progress|consultation",
            "icd10_codes": [
                {
                    "code": "code",
                    "description": "description",
                    "confidence": 0.95,
                    "primary": true,
                    "evidence": {
                        "direct_quotes": ["relevant text from note supporting this code"],
                        "reasoning": "Detailed explanation of why this code was selected",
                        "guidelines_applied": ["specific coding guidelines used"]
                    }
                }
            ],
            "cpt_codes": [
                {
                    "code": "code",
                    "description": "description",
                    "confidence": 0.7,
                    "evidence": {
                        "direct_quotes": ["relevant text from note supporting this code"],
                        "reasoning": "Detailed explanation of why this code was selected",
                        "guidelines_applied": ["specific coding guidelines used"]
                    },
                    "alternative_codes": [
                        {
                            "code": "alternative_cpt",
                            "description": "description",
                            "required_documentation": "specific missing documentation needed",
                            "why_considered": "explanation of why this is a potential alternative"
                        }
                    ]
                }
            ],
            "documentation_gaps": [
                {
                    "severity": "critical|moderate|minor",
                    "description": "Specific missing or ambiguous element",
                    "impact": {
                        "affected_codes": ["codes impacted by this gap"],
                        "current_limitation": "How this affects current code selection",
                        "potential_improvement": "What better codes could be used with proper documentation"
                    },
                    "recommendation": {
                        "what_to_add": "Specific guidance on documentation needed",
                        "example": "Example of proper documentation",
                        "rationale": "Why this documentation is important"
                    }
                }
            ]
        }

        Guidelines:
        1. Response must be valid JSON:
           - No trailing commas
           - All property names must be in double quotes
           - No JavaScript comments
           - No undefined or NaN values
           - All strings must be in double quotes
           - Numbers should be plain (no quotes)
           - Boolean values should be true/false (no quotes)

        2. Always provide clear evidence trail:
           - Quote specific text that supports each code
           - Explain reasoning for code selection
           - Reference specific coding guidelines used

        3. For documentation gaps:
           - Critical: Missing elements that prevent proper code assignment
           - Moderate: Elements that could change code selection
           - Minor: Elements that would strengthen coding but don't change selection

        4. Confidence scoring:
           - High (0.9-1.0): Complete, unambiguous documentation
           - Medium (0.7-0.8): Some minor missing elements
           - Low (0.5-0.6): Significant documentation gaps
           - Any ambiguity caps confidence at 0.7

        5. Alternative codes:
           - Only suggest when documentation indicates possible alternatives
           - Maximum 3 alternatives per code
           - Must explain what documentation would support each alternative

        6. Length-appropriate analysis:
           - Brief notes: Focus on documented elements only
           - Comprehensive notes: Systematic analysis of all sections
           - Always maintain same level of evidence trail regardless of note length

        IMPORTANT: Return response as a valid JSON object only. Do not include any explanatory text before or after the JSON object. Do not use markdown formatting or code blocks. The response should be a pure JSON object that can be parsed directly.
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
            'note_type': str(result.get('note_type', 'brief')).lower(),
            'icd10_codes': [],
            'cpt_codes': [],
            'documentation_gaps': []
        }

        # Validate ICD-10 codes
        if 'icd10_codes' in result and isinstance(result['icd10_codes'], list):
            for code in result['icd10_codes']:
                try:
                    validated_code = {
                        'code': str(code.get('code', '')).strip(),
                        'description': str(code.get('description', '')).strip(),
                        'confidence': max(0.0, min(1.0, float(code.get('confidence', 0.5)))),
                        'primary': bool(code.get('primary', False)),
                        'evidence': {
                            'direct_quotes': code.get('evidence', {}).get('direct_quotes', []),
                            'reasoning': str(code.get('evidence', {}).get('reasoning', '')),
                            'guidelines_applied': code.get('evidence', {}).get('guidelines_applied', [])
                        }
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
                        'evidence': {
                            'direct_quotes': code.get('evidence', {}).get('direct_quotes', []),
                            'reasoning': str(code.get('evidence', {}).get('reasoning', '')),
                            'guidelines_applied': code.get('evidence', {}).get('guidelines_applied', [])
                        },
                        'alternative_codes': code.get('alternative_codes', [])
                    }
                    if validated_code['code']:  # Only add if code is not empty
                        validated_result['cpt_codes'].append(validated_code)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping invalid CPT code entry: {e}")
                    continue

        # Validate documentation gaps
        if 'documentation_gaps' in result and isinstance(result['documentation_gaps'], list):
            for gap in result['documentation_gaps']:
                try:
                    validated_gap = {
                        'severity': str(gap.get('severity', 'minor')).lower(),
                        'description': str(gap.get('description', '')),
                        'impact': gap.get('impact', {
                            'affected_codes': [],
                            'current_limitation': '',
                            'potential_improvement': ''
                        }),
                        'recommendation': gap.get('recommendation', {
                            'what_to_add': '',
                            'example': '',
                            'rationale': ''
                        })
                    }
                    validated_result['documentation_gaps'].append(validated_gap)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping invalid documentation gap entry: {e}")
                    continue

        return validated_result
