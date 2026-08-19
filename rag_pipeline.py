"""
RAG Pipeline Module.
Handles transcript chunking, embedding generation using the native Ollama Python SDK
(`granite-embedding:30m`), indexing in ChromaDB, and static retrieval of the most relevant chunks.
"""

from typing import List, Dict, Any, Optional
import uuid
import re
import ollama
import chromadb
import config


class TranscriptRAG:
    """
    RAG engine for chunking conversation transcripts, generating embeddings via Ollama,
    indexing into ChromaDB, and retrieving relevant context for downstream extraction.
    """

    # Static queries tailored to extract the 7 target features
    STATIC_QUERIES = [
        "Customer name, contact phone number, lead source, Instagram, Meta Ads marketing campaign",
        "Business requirements, product discussion, pricing plans, pain points and objections",
        "Closing agreed next steps, follow up date, scheduled demo call slot, case studies walkthrough"
    ]

    def __init__(
        self,
        embedding_model: str = None,
        ollama_host: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        """
        Initialize the RAG pipeline.

        :param embedding_model: Ollama embedding model name.
        :param ollama_host: URL of the Ollama server.
        :param chunk_size: Approximate size of each chunk (characters).
        :param chunk_overlap: Overlap between consecutive chunks.
        """
        self.embedding_model = embedding_model or config.OLLAMA_EMBED_MODEL
        self.ollama_host = ollama_host or config.OLLAMA_HOST
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

        # Initialize Ollama client
        self.ollama_client = ollama.Client(host=self.ollama_host)
        
        # Initialize in-memory ChromaDB client
        self.chroma_client = chromadb.Client()

    def chunk_transcript(self, text: str) -> List[str]:
        """
        Splits a conversation transcript into contextual chunks.
        Handles speaker turns (e.g., [CUSTOMER], [SALES_REP] or CUSTOMER:, SALES_REP:)
        and creates overlapping segments.

        :param text: Full transcript text.
        :return: List of chunk strings.
        """
        if not text or not text.strip():
            return []

        # Split on turn boundaries using non-capturing lookahead
        raw_parts = re.split(r"(?=(?:\[CUSTOMER\]|\[SALES_REP\]|CUSTOMER:|SALES_REP:))", text)
        clean_turns = [p.strip() for p in raw_parts if p and p.strip()]

        # If turn splitting didn't yield multiple segments, fall back to sentence/sliding window
        if len(clean_turns) <= 1:
            raw_text = text.strip()
            chunks = []
            start = 0
            while start < len(raw_text):
                end = min(start + self.chunk_size, len(raw_text))
                # Try not to break words in the middle
                if end < len(raw_text):
                    next_space = raw_text.rfind(" ", start, end)
                    if next_space > start:
                        end = next_space
                chunk = raw_text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                if end >= len(raw_text):
                    break
                start = max(start + 1, end - self.chunk_overlap)
            return chunks

        # Group conversation turns into chunks of ~chunk_size
        chunks = []
        current_chunk = []
        current_len = 0

        for turn in clean_turns:
            turn_len = len(turn)
            if current_len + turn_len > self.chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                # Keep last turn for overlap if possible
                if len(current_chunk) > 1:
                    current_chunk = [current_chunk[-1], turn]
                    current_len = len(current_chunk[0]) + turn_len
                else:
                    current_chunk = [turn]
                    current_len = turn_len
            else:
                current_chunk.append(turn)
                current_len += turn_len

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate vector embeddings using Ollama Python SDK.
        Tries configured model first, with automatic fallback if necessary.

        :param texts: List of text strings to embed.
        :return: List of float embedding vectors.
        """
        if not texts:
            return []

        models_to_try = [self.embedding_model] + [
            m for m in config.FALLBACK_EMBED_MODELS if m != self.embedding_model
        ]

        last_error = None
        for model in models_to_try:
            try:
                # Use native ollama.embed()
                response = self.ollama_client.embed(
                    model=model,
                    input=texts
                )
                self.embedding_model = model  # Remember working model name
                return response["embeddings"]
            except Exception as e:
                last_error = e
                continue

        raise RuntimeError(
            f"Failed to generate embeddings using Ollama models {models_to_try}. "
            f"Please ensure Ollama is running (`ollama serve`) and the model is pulled (`ollama pull granite-embedding:30m`). "
            f"Error: {last_error}"
        )

    def retrieve_context(
        self,
        transcript_text: str,
        collection_prefix: str = "transcript",
        top_k: int = 5,
        use_multi_query: bool = True
    ) -> List[str]:
        """
        Full RAG flow:
        1. Chunks the transcript.
        2. Generates embeddings using Ollama.
        3. Indexes chunks into a temporary ChromaDB collection.
        4. Executes static retrieval queries.
        5. Returns deduplicated top chunks.

        :param transcript_text: Full raw transcript string.
        :param collection_prefix: Prefix for temporary ChromaDB collection.
        :param top_k: Maximum number of chunks to return.
        :param use_multi_query: Whether to run 3 static queries or a single query.
        :return: List of retrieved context chunk strings.
        """
        chunks = self.chunk_transcript(transcript_text)
        if not chunks:
            return []

        # If transcript is short, return all chunks directly
        if len(chunks) <= 3:
            return chunks

        # Compute embeddings for all chunks via Ollama
        chunk_embeddings = self.get_embeddings(chunks)
        chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]

        # Create isolated collection in ChromaDB
        collection_name = f"{collection_prefix}_{uuid.uuid4().hex[:8]}"
        collection = self.chroma_client.create_collection(name=collection_name)

        try:
            # Add to ChromaDB
            collection.add(
                documents=chunks,
                embeddings=chunk_embeddings,
                ids=chunk_ids
            )

            retrieved_chunks = []
            seen_chunks = set()

            if use_multi_query:
                # Static flow: execute each targeted query to ensure balanced coverage
                per_query_limit = 2
                for query_text in self.STATIC_QUERIES:
                    q_embed = self.get_embeddings([query_text])[0]
                    query_res = collection.query(
                        query_embeddings=[q_embed],
                        n_results=min(per_query_limit, len(chunks))
                    )
                    docs = query_res.get("documents", [[]])[0]
                    for doc in docs:
                        if doc not in seen_chunks:
                            seen_chunks.add(doc)
                            retrieved_chunks.append(doc)
            else:
                # Single consolidated static query
                consolidated_query = (
                    "Customer name, phone number, lead source, last discussed topic, "
                    "interest status, assigned sales rep, and follow up details."
                )
                q_embed = self.get_embeddings([consolidated_query])[0]
                query_res = collection.query(
                    query_embeddings=[q_embed],
                    n_results=min(top_k, len(chunks))
                )
                docs = query_res.get("documents", [[]])[0]
                retrieved_chunks = docs

            return retrieved_chunks[:max(top_k, len(retrieved_chunks))]

        finally:
            # Clean up ChromaDB collection to free memory
            try:
                self.chroma_client.delete_collection(name=collection_name)
            except Exception:
                pass


if __name__ == "__main__":
    from pathlib import Path
    sample_file = Path("./transcripts/transcript.txt")
    if sample_file.exists():
        with open(sample_file, "r", encoding="utf-8") as f:
            text = f.read()

        rag = TranscriptRAG()
        print(f"Original Text Length: {len(text)} chars")
        chunks = rag.chunk_transcript(text)
        print(f"Total Chunks Generated: {len(chunks)}")
        
        print("\nRetrieving context via Ollama embeddings + ChromaDB...")
        top_chunks = rag.retrieve_context(text, top_k=3)
        print(f"Retrieved {len(top_chunks)} chunks:")
        for idx, ch in enumerate(top_chunks, 1):
            print(f"\n--- Chunk {idx} ---")
            print(ch)
    else:
        print("Sample transcript file not found.")
