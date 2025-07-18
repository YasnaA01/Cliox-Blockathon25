from logging import getLogger
from pathlib import Path
from typing import Any, Optional, TypeVar
from oceanprotocol_job_details.ocean import JobDetails

# =============================== IMPORT LIBRARY ====================
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import requests
import json
# =============================== END ===============================

T = TypeVar("T")
logger = getLogger(__name__)

class Algorithm:
    # TODO: [optional] add class variables here

    def __init__(self, job_details: JobDetails):
        self._job_details = job_details
        self.results: Optional[Any] = None

    def _validate_input(self) -> None:
        if not self._job_details.files:
            logger.warning("No files found")
            raise ValueError("No files found")

    def run(self) -> "Algorithm":
        # TODO: 1. Initialize results type 
        self.results = {}

        # TODO: 2. validate input here
        self._validate_input()

        # TODO: 3. get input files here
        input_files = self._job_details.files.files[0].input_files
        question_file = str(input_files[0])  # expects question.txt
        enron_csv_path = "enron.csv"         # assumes dataset is already mounted

        # TODO: 4. run algorithm here
        try:
            # 1. Load question
            with open(question_file, "r", encoding="utf-8") as f:
                question = f.read().strip()

            # 2. Load Enron dataset
            df = pd.read_csv(enron_csv_path)
            texts = df["message"].dropna().tolist()

            # 3. Split into chunks
            chunk_size = 500
            chunks = []
            for text in texts:
                chunks.extend([text[i:i+chunk_size] for i in range(0, len(text), chunk_size)])

            # 4. Embed chunks
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            chunk_embeddings = embedder.encode(chunks)
            index = faiss.IndexFlatL2(chunk_embeddings[0].shape[0])
            index.add(chunk_embeddings)

            # 5. Embed question and search top chunks
            question_embedding = embedder.encode([question])
            _, indices = index.search(question_embedding, 3)
            top_chunks = [chunks[i] for i in indices[0]]

            # 6. Ask local LLM (Mistral via Ollama)
            prompt = f"Beantworte die folgende Frage basierend auf den E-Mails:\n\n"
            prompt += "\n\n".join(top_chunks)
            prompt += f"\n\nFrage: {question}"

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "mistral", "prompt": prompt, "stream": False}
            )

            answer = response.json().get("response", "Keine Antwort erhalten.")

            # TODO: 5. save results here
            self.results = {
                "question": question,
                "answer": answer,
                "sources": top_chunks
            }

        except Exception as e:
            logger.exception(f"Fehler während der Verarbeitung: {e}")
            self.results = {"error": str(e)}

        # TODO: 6. return self
        return self

    def save_result(self, path: Path) -> None:
        # TODO: 7. define/add result path here
        result_path = path / "result.json"

        with open(result_path, "w", encoding="utf-8") as f:
            try:
                # TODO: 8. save results here
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.exception(f"Error saving data: {e}")
