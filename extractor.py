"""
Groq Lead Information Extractor Module.
Uses Groq's `openai/gpt-oss-20b` model in native JSON mode to strictly extract
the 7 target features from retrieved transcript chunks with zero hallucination.
"""

from typing import Dict, Any, List, Optional
import json
import os
from groq import Groq
import config


class LeadExtractor:
    """
    Extracts structured sales lead intelligence from conversation context
    using Groq's LLM with JSON mode.
    """

    SYSTEM_PROMPT = """You are an accurate, strict, zero-assumption sales data extraction assistant.
Your task is to extract exactly 7 specific fields from the provided conversation transcript chunks.

Output format MUST be a valid JSON object with EXACTLY these keys:
{
  "Name": "<Customer's full or first name if mentioned>",
  "Phone no.": "<Customer's phone number if mentioned>",
  "Source": "<Marketing/lead source, e.g. Instagram, Meta ads, Website, Comment, etc.>",
  "Last Discussed topic": "<Concise factual summary of the last topic discussed>",
  "Status": "<Current lead status, e.g. Fresh, Interested, Demo Scheduled, Follow-up Pending, etc.>",
  "Assigned to": "<Sales representative name handling the lead>",
  "Follow up?": "<Follow-up plan/commitment or date/time if mentioned>"
}

CRITICAL RULES:
1. Extract ONLY information explicitly and factually stated in the text.
2. If any field is NOT found or is unclear, leave its value as an empty string ("").
3. DO NOT guess, assume, infer, extrapolate, or fabricate any information.
4. DO NOT add any extra keys to the JSON.
5. Return ONLY the JSON object.
"""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the Groq Lead Extractor.

        :param api_key: Groq API Key. If None, reads from config/environment.
        :param model: Groq model name. Defaults to `openai/gpt-oss-20b`.
        """
        self.api_key = api_key or config.GROQ_API_KEY
        self.model = model or config.GROQ_MODEL
        self._client: Optional[Groq] = None

    @property
    def client(self) -> Groq:
        """Lazy initialization of the Groq client."""
        if self._client is None:
            if not self.api_key:
                # Re-check environment in case loaded later
                self.api_key = os.getenv("GROQ_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "GROQ_API_KEY is not set. Please add `GROQ_API_KEY=your_key` to your `.env` file "
                    "or export it as an environment variable."
                )
            self._client = Groq(api_key=self.api_key)
        return self._client

    def extract(self, context_chunks: List[str]) -> Dict[str, str]:
        """
        Extract the 7 target features from a list of retrieved transcript chunks.

        :param context_chunks: List of relevant transcript text chunks.
        :return: Clean dictionary containing all 7 target features.
        """
        if not context_chunks:
            return {k: "" for k in config.TARGET_SCHEMA}

        context_text = "\n\n---\n\n".join(context_chunks)
        user_message = f"Here is the context from the conversation transcript:\n\n{context_text}\n\nExtract the required 7 fields in JSON format."

        try:
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )

            raw_response = chat_completion.choices[0].message.content
            parsed_data = json.loads(raw_response)

            # Ensure all required 7 fields exist and empty strings if missing
            sanitized_result = {}
            for key in config.TARGET_SCHEMA:
                val = parsed_data.get(key, "")
                sanitized_result[key] = str(val).strip() if val is not None else ""

            return sanitized_result

        except json.JSONDecodeError as e:
            print(f"[LeadExtractor] JSON Decode Error: {e}. Raw content: {raw_response}")
            # Fallback schema with empty values
            return {k: "" for k in config.TARGET_SCHEMA}
        except Exception as e:
            print(f"[LeadExtractor] Error calling Groq API: {e}")
            raise


if __name__ == "__main__":
    from pathlib import Path
    extractor = LeadExtractor()
    sample_context = [
        "[CUSTOMER]Hello?[SALES_REP]Hi, am I speaking with Rohan Mehta?[CUSTOMER]Yes, speaking. Who is this?[SALES_REP]Hi Rohan, this is Priya calling from LeadFlow. You had commented on one of our Instagram posts last week about automating your customer follow-ups.",
        "[SALES_REP]Perfect. I'll follow up on WhatsApp by tomorrow evening. If you can share even a tentative slot, I'll block it. And in the meantime I'm sending the case studies and a short video walkthrough.[CUSTOMER]Sounds good. Thanks Priya."
    ]
    try:
        res = extractor.extract(sample_context)
        print("Extraction Result:")
        print(json.dumps(res, indent=2))
    except Exception as err:
        print(f"Extraction test error: {err}")
