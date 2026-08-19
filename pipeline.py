"""
Lead Generation End-to-End Orchestrator Pipeline.
Connects Classifier, RAG Engine, Groq Extractor, and Excel Manager
in a modular, plug-and-play architecture with support for concurrent processing.
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import config
from classifier import LeadClassifier
from rag_pipeline import TranscriptRAG
from extractor import LeadExtractor
from excel_manager import ExcelManager


class LeadGenPipeline:
    """
    Plug-and-play end-to-end pipeline for lead classification,
    RAG-based intelligence extraction, and Excel persistence.
    """

    def __init__(
        self,
        classifier: Optional[LeadClassifier] = None,
        rag_engine: Optional[TranscriptRAG] = None,
        extractor: Optional[LeadExtractor] = None,
        excel_manager: Optional[ExcelManager] = None,
    ):
        """
        Initialize pipeline components. Any component can be customized or substituted.
        """
        self.classifier = classifier or LeadClassifier()
        self.rag_engine = rag_engine or TranscriptRAG()
        self.extractor = extractor or LeadExtractor()
        self.excel_manager = excel_manager or ExcelManager()

    def process_transcript(
        self,
        transcript_text: str,
        identifier: str = "transcript",
        save_to_excel: bool = True
    ) -> Dict[str, Any]:
        """
        Execute the full pipeline on a single transcript:
        1. Classify transcript (0 = Not a lead, 1 = Potential lead).
        2. If 1:
           a. Chunk & embed into ChromaDB via Ollama.
           b. Retrieve top relevant context chunks via static query flow.
           c. Extract target 7 features using Groq LLM (JSON mode).
           d. Save to Excel workbook.

        :param transcript_text: Raw conversation transcript string.
        :param identifier: Name or ID of the transcript for logging.
        :param save_to_excel: Whether to append positive leads to Excel.
        :return: Result dictionary with classification details and extracted JSON.
        """
        start_time = time.time()
        print(f"\n[Pipeline] Processing '{identifier}'...")

        # Step 1: ML Model Classification
        class_res = self.classifier.predict(transcript_text)
        prediction = class_res["prediction"]
        confidence = class_res["confidence"]

        if prediction == 0:
            elapsed = time.time() - start_time
            print(f"[Pipeline] [X] '{identifier}' classified as NOT a potential lead (Confidence: {confidence:.2%}). Skipping extraction.")
            return {
                "identifier": identifier,
                "status": "skipped",
                "is_lead": False,
                "prediction": 0,
                "confidence": confidence,
                "probabilities": class_res["probabilities"],
                "retrieved_chunks": [],
                "extracted_data": None,
                "elapsed_seconds": round(elapsed, 2)
            }

        print(f"[Pipeline] [OK] '{identifier}' classified as POTENTIAL LEAD (Confidence: {confidence:.2%}). Proceeding to RAG extraction...")

        # Step 2: RAG Context Retrieval (Ollama + ChromaDB)
        retrieved_chunks = self.rag_engine.retrieve_context(
            transcript_text=transcript_text,
            collection_prefix=identifier.replace(" ", "_"),
            top_k=config.TOP_K_CHUNKS,
            use_multi_query=True
        )
        print(f"[Pipeline] [RAG] Retrieved {len(retrieved_chunks)} relevant context chunks.")

        # Step 3: Structured Information Extraction (Groq openai/gpt-oss-20b)
        extracted_data = self.extractor.extract(retrieved_chunks)
        print(f"[Pipeline] [LLM] Information extracted successfully via Groq.")

        # Step 4: Excel Storage
        if save_to_excel:
            self.excel_manager.append_lead(extracted_data)
            print(f"[Pipeline] [Excel] Saved lead record into '{self.excel_manager.file_path}'.")

        elapsed = time.time() - start_time
        return {
            "identifier": identifier,
            "status": "success",
            "is_lead": True,
            "prediction": 1,
            "confidence": confidence,
            "probabilities": class_res["probabilities"],
            "retrieved_chunks": retrieved_chunks,
            "extracted_data": extracted_data,
            "elapsed_seconds": round(elapsed, 2)
        }

    def process_file(self, file_path: Union[str, Path], save_to_excel: bool = True) -> Dict[str, Any]:
        """
        Process a single transcript file from disk.

        :param file_path: Path to the transcript text file.
        :param save_to_excel: Whether to append positive leads to Excel.
        :return: Result dictionary.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Transcript file not found: {path}")

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        return self.process_transcript(
            transcript_text=content,
            identifier=path.name,
            save_to_excel=save_to_excel
        )

    def process_batch_concurrently(
        self,
        transcripts: List[Dict[str, str]],
        max_workers: int = None,
        save_to_excel: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process multiple transcripts concurrently using a thread pool.

        :param transcripts: List of dicts with keys {"identifier": str, "text": str}.
        :param max_workers: Number of concurrent workers. Defaults to `config.MAX_CONCURRENT_WORKERS`.
        :param save_to_excel: Whether to save extracted leads to Excel.
        :return: List of result dictionaries.
        """
        workers = max_workers or config.MAX_CONCURRENT_WORKERS
        results = []

        print(f"\n[Pipeline] Starting concurrent batch execution for {len(transcripts)} transcripts with {workers} workers...")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_item = {
                executor.submit(
                    self.process_transcript,
                    item["text"],
                    item.get("identifier", f"item_{idx}"),
                    save_to_excel
                ): item
                for idx, item in enumerate(transcripts)
            }

            for future in as_completed(future_to_item):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    item = future_to_item[future]
                    ident = item.get("identifier", "unknown")
                    print(f"[Pipeline] Error processing '{ident}': {e}")
                    results.append({
                        "identifier": ident,
                        "status": "error",
                        "error": str(e)
                    })

        return results

    def process_directory_concurrently(
        self,
        directory_path: Union[str, Path],
        max_workers: int = None,
        save_to_excel: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Scan a directory for all .txt transcript files and process them concurrently.

        :param directory_path: Path to directory containing transcript files.
        :param max_workers: Number of concurrent workers.
        :param save_to_excel: Whether to save positive leads to Excel.
        :return: List of results for each file.
        """
        dir_p = Path(directory_path)
        if not dir_p.exists() or not dir_p.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_p}")

        txt_files = list(dir_p.glob("*.txt"))
        if not txt_files:
            print(f"[Pipeline] No .txt files found in directory: {dir_p}")
            return []

        batch_items = []
        for f in txt_files:
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                batch_items.append({
                    "identifier": f.name,
                    "text": fh.read()
                })

        return self.process_batch_concurrently(
            transcripts=batch_items,
            max_workers=max_workers,
            save_to_excel=save_to_excel
        )
