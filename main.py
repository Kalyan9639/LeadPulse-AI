"""
Main Entry Point for Lead Generation and Intelligence Extraction Pipeline.
Provides CLI and execution orchestration.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import config
from classifier import LeadClassifier
from rag_pipeline import TranscriptRAG
from extractor import LeadExtractor
from excel_manager import ExcelManager
from pipeline import LeadGenPipeline


def print_banner():
    print("=" * 70)
    print("      AI Lead Generation & Intelligence Extraction Pipeline      ")
    print("=" * 70)


def print_single_result(result: dict):
    print("\n" + "-" * 70)
    print(f"TRANSCRIPT: {result['identifier']}")
    print(f"STATUS    : {result['status'].upper()}")
    print(f"IS LEAD   : {result.get('is_lead')}")
    print(f"CONFIDENCE: {result.get('confidence', 0.0):.2%}")
    print(f"TIME TAKEN: {result.get('elapsed_seconds', 0.0)}s")
    print("-" * 70)

    if result.get("is_lead") and result.get("extracted_data"):
        data = result["extracted_data"]
        print("\nExtracted Lead Intelligence (JSON):")
        print(json.dumps(data, indent=2, ensure_ascii=False))

        print("\nFeatures Summary Table:")
        max_key_len = max(len(k) for k in config.TARGET_SCHEMA)
        for key in config.TARGET_SCHEMA:
            val = data.get(key, "")
            display_val = val if val else "[NOT MENTIONED / EMPTY]"
            print(f"  - {key.ljust(max_key_len)} : {display_val}")
    print("-" * 70 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lead Classification, RAG Extraction, and Excel Export Pipeline."
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        default="./transcripts/transcript.txt",
        help="Path to a single transcript .txt file (default: ./transcripts/transcript.txt)"
    )
    parser.add_argument(
        "-d", "--dir",
        type=str,
        default=None,
        help="Path to a directory containing multiple .txt transcript files for batch processing."
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=config.MAX_CONCURRENT_WORKERS,
        help=f"Number of concurrent workers for batch processing (default: {config.MAX_CONCURRENT_WORKERS})"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=config.DEFAULT_EXCEL_PATH,
        help=f"Output Excel file path (default: {config.DEFAULT_EXCEL_PATH})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional local path to a .joblib model file (if not downloading from Hugging Face)"
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=config.HF_REPO_ID,
        help=f"Hugging Face model repository ID (default: {config.HF_REPO_ID})"
    )
    parser.add_argument(
        "--hf-file",
        type=str,
        default=config.HF_MODEL_FILENAME,
        help=f"Model filename in Hugging Face repository (default: {config.HF_MODEL_FILENAME})"
    )
    parser.add_argument(
        "--no-excel",
        action="store_true",
        help="Skip saving extracted leads to Excel"
    )
    return parser.parse_args()


def main():
    print_banner()
    args = parse_args()

    # Initialize components
    classifier = LeadClassifier(
        model_path=args.model,
        repo_id=args.hf_repo,
        filename=args.hf_file
    )
    rag_engine = TranscriptRAG()
    excel_manager = ExcelManager(file_path=args.output)
    
    # Initialize extractor (Groq)
    try:
        extractor = LeadExtractor()
    except Exception as e:
        print(f"[Warning] Extractor initialized with notice: {e}")
        extractor = LeadExtractor()

    pipeline = LeadGenPipeline(
        classifier=classifier,
        rag_engine=rag_engine,
        extractor=extractor,
        excel_manager=excel_manager
    )

    save_excel = not args.no_excel

    if args.dir:
        # Directory batch mode (concurrent)
        dir_path = Path(args.dir)
        print(f"[Mode] Directory Batch Processing: {dir_path}")
        results = pipeline.process_directory_concurrently(
            directory_path=dir_path,
            max_workers=args.workers,
            save_to_excel=save_excel
        )

        leads_count = sum(1 for r in results if r.get("is_lead"))
        skipped_count = sum(1 for r in results if r.get("status") == "skipped")
        error_count = sum(1 for r in results if r.get("status") == "error")

        print("\n" + "=" * 70)
        print("                   BATCH EXECUTION SUMMARY                   ")
        print("=" * 70)
        print(f"Total Processed : {len(results)}")
        print(f"Potential Leads : {leads_count}")
        print(f"Skipped Non-Leads: {skipped_count}")
        print(f"Errors          : {error_count}")
        if save_excel and leads_count > 0:
            print(f"Excel Output    : {args.output}")
        print("=" * 70)

    else:
        # Single file mode
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: Transcript file not found at '{file_path}'")
            sys.exit(1)

        result = pipeline.process_file(file_path=file_path, save_to_excel=save_excel)
        print_single_result(result)


if __name__ == "__main__":
    main()
