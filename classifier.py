"""
Lead Classifier Module.
Uses the trained scikit-learn model to determine whether a conversation
transcript can be converted into a potential sales lead (1) or not (0).
"""

from typing import Dict, Any, List, Union
from pathlib import Path
import joblib
import config


class LeadClassifier:
    """
    Classifies customer-sales conversation transcripts using a pretrained
    scikit-learn pipeline downloaded at runtime from Hugging Face.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = None,
        repo_id: str = None,
        filename: str = None
    ):
        """
        Initialize the classifier with a model downloaded from Hugging Face or from a local path.

        :param model_path: Optional explicit local path to a .joblib model file.
        :param repo_id: Hugging Face repository ID (default: 'mr-checker/SalesGen').
        :param filename: Model filename in the repo (default: 'sales_model_balanced.joblib').
        """
        self.model_path = model_path
        self.repo_id = repo_id or config.HF_REPO_ID
        self.filename = filename or config.HF_MODEL_FILENAME
        self._load_model()

    def _load_model(self) -> None:
        """Loads the model pipeline, downloading from Hugging Face if needed."""
        target_path = None

        # 1. If explicit local file path provided and exists
        if self.model_path and Path(self.model_path).exists():
            target_path = Path(self.model_path)
        else:
            # 2. Download model at runtime from Hugging Face repo
            try:
                from huggingface_hub import hf_hub_download
                print(f"[LeadClassifier] Downloading model at runtime from Hugging Face ({self.repo_id}/{self.filename})...")
                downloaded_file = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=self.filename
                )
                target_path = Path(downloaded_file)
                print(f"[LeadClassifier] Successfully fetched model from Hugging Face: {target_path}")
            except Exception as hf_err:
                print(f"[LeadClassifier] Hugging Face download notice: {hf_err}")
                # 3. Fallback to local copy if available
                fallback = Path(config.FALLBACK_LOCAL_MODEL)
                if fallback.exists():
                    print(f"[LeadClassifier] Falling back to local model at: '{fallback}'")
                    target_path = fallback
                else:
                    raise RuntimeError(f"Could not download model from Hugging Face and no local fallback found: {hf_err}")

        try:
            self.model = joblib.load(str(target_path))
            self.model_path = str(target_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {target_path}: {e}")

    def predict(self, transcript_text: str) -> Dict[str, Any]:
        """
        Predict whether a single transcript is a potential lead.

        :param transcript_text: Full string conversation transcript.
        :return: Dictionary containing prediction (0 or 1), is_lead (bool),
                 confidence score, and raw class probabilities.
        """
        if not transcript_text or not transcript_text.strip():
            return {
                "prediction": 0,
                "is_lead": False,
                "confidence": 0.0,
                "probabilities": [1.0, 0.0]
            }

        input_data = [transcript_text]
        pred = int(self.model.predict(input_data)[0])
        
        # Check if predict_proba is supported
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(input_data)[0].tolist()
            confidence = float(probs[pred])
        else:
            probs = [1.0 - float(pred), float(pred)]
            confidence = 1.0

        return {
            "prediction": pred,
            "is_lead": pred == 1,
            "confidence": confidence,
            "probabilities": probs
        }

    def predict_batch(self, transcripts: List[str]) -> List[Dict[str, Any]]:
        """
        Batch prediction for multiple transcripts.

        :param transcripts: List of transcript strings.
        :return: List of prediction dictionaries.
        """
        if not transcripts:
            return []

        preds = self.model.predict(transcripts)
        has_proba = hasattr(self.model, "predict_proba")
        probs = self.model.predict_proba(transcripts) if has_proba else None

        results = []
        for i, pred in enumerate(preds):
            pred_int = int(pred)
            if has_proba:
                prob_list = probs[i].tolist()
                conf = float(prob_list[pred_int])
            else:
                prob_list = [1.0 - float(pred_int), float(pred_int)]
                conf = 1.0

            results.append({
                "prediction": pred_int,
                "is_lead": pred_int == 1,
                "confidence": conf,
                "probabilities": prob_list
            })

        return results


if __name__ == "__main__":
    import sys
    classifier = LeadClassifier()
    sample_file = Path("./transcripts/transcript.txt")
    if sample_file.exists():
        with open(sample_file, "r", encoding="utf-8") as f:
            sample_text = f.read()
        res = classifier.predict(sample_text)
        print(f"Sample File: {sample_file}")
        print(f"Prediction: {res['prediction']} (Lead: {res['is_lead']})")
        print(f"Confidence: {res['confidence']:.4f}")
        print(f"Probabilities: {res['probabilities']}")
    else:
        print("Sample transcript file not found.")
