from __future__ import annotations

import hashlib
import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# Expected location: <project_root>/src/rag_assistant/ingest.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAME = "banking_knowledge"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
BATCH_SIZE = 100

def relative_source(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()

def load_pdf(path: Path) -> list[Document]:
    """Load a PDF page-by-page to preserve page citations."""
    source = relative_source(path)
    category = path.relative_to(DOCUMENTS_DIR).parts[0]
    loaded = []

    for page_number, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            loaded.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": source,
                        "page": page_number,
                        "category": category,
                        "document_type": "pdf",
                    },
                )
            )
    return loaded

def fee_record_to_text(record: dict) -> str:
    """Build searchable text from category, section, headers, and values."""
    category = str(record.get("category") or "").strip()
    section = str(record.get("section") or "Ürün ve hizmet ücretleri").strip()
    headers = record.get("headers") or []
    values = record.get("values") or []

    details = [
        f"{header}: {value}"
        for header, value in zip(headers, values)
        if value is not None and str(value).strip()
    ]

    hierarchy = []
    if category:
        hierarchy.append(f"Ana kategori: {category}")
    hierarchy.append(f"Bölüm: {section}")

    return "\\n".join([*hierarchy, *details])

def load_jsonl(path: Path) -> list[Document]:
    """Load each JSONL row as its own document, preserving fee-table metadata."""
    source = relative_source(path)
    category = path.relative_to(DOCUMENTS_DIR).parts[0]
    loaded = []

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {source}"
                ) from exc

            text = fee_record_to_text(record)
            if not text.strip():
                continue

            record_category = str(record.get("category") or category).strip()
            record_section = str(record.get("section") or "").strip()

            loaded.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": source,
                        "category": record_category,
                        "document_type": "jsonl",
                        "record": line_number,
                        "section": record_section,
                        "table_index": int(record.get("table_index", 0)),
                        "row_index": int(record.get("row_index", 0)),
                        "source_url": str(record.get("source_url", "")),
                    },
                )
            )
    return loaded

def load_documents() -> list[Document]:
    """Recursively ingest PDFs and JSONL files from data/documents.

    The Akbank archive has HTML, Markdown and JSONL representations of the
    same fee tables. JSONL is the canonical ingest format for this dataset;
    HTML and Markdown are skipped to avoid indexing duplicate copies.
    """
    if not DOCUMENTS_DIR.is_dir():
        raise FileNotFoundError(f"Documents directory not found: {DOCUMENTS_DIR}")

    documents: list[Document] = []
    pdf_paths = sorted(DOCUMENTS_DIR.rglob("*.pdf"))
    jsonl_paths = sorted(DOCUMENTS_DIR.rglob("*.jsonl"))

    for path in pdf_paths:
        try:
            docs = load_pdf(path)
        except Exception as exc:
            raise RuntimeError(f"Could not read PDF {relative_source(path)}") from exc
        documents.extend(docs)
        print(f"[LOAD] PDF {relative_source(path)}: {len(docs)} non-empty pages")

    for path in jsonl_paths:
        docs = load_jsonl(path)
        documents.extend(docs)
        print(f"[LOAD] JSONL {relative_source(path)}: {len(docs)} records")

    if not documents:
        raise RuntimeError(f"No supported PDF/JSONL content found in {DOCUMENTS_DIR}")
    return documents

def stable_chunk_id(document: Document) -> str:
    identity = {
        "source": document.metadata.get("source"),
        "page": document.metadata.get("page"),
        "record": document.metadata.get("record"),
        "start_index": document.metadata.get("start_index", 0),
        # "text": document.page_content,
    }
    serialized = json.dumps(identity, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

def main() -> None:
    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise RuntimeError("No chunks were produced from the loaded documents.")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # Rebuild this demo collection from the current archive on each run.
    # This removes stale chunks when a source file is removed or changed.
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
    vector_store.delete_collection()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )

    ids = [stable_chunk_id(chunk) for chunk in chunks]
    for start in range(0, len(chunks), BATCH_SIZE):
        vector_store.add_documents(
            documents=chunks[start: start + BATCH_SIZE],
            ids=ids[start: start + BATCH_SIZE],
        )

    print("\nIngestion complete")
    print(f"Documents/pages/records loaded: {len(documents)}")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Chroma directory: {CHROMA_DIR}")

if __name__ == "__main__":
    main()
