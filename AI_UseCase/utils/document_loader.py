"""
Document Loading & Chunking
Handles PDF, DOCX, and TXT files with intelligent chunking.
"""

import os
import logging
import tempfile
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def load_uploaded_documents(uploaded_files):
    """
    Load documents from Streamlit uploaded file objects.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects

    Returns:
        List of LangChain Document objects
    """
    documents = []

    for uploaded_file in uploaded_files:
        try:
            file_name = uploaded_file.name
            file_extension = os.path.splitext(file_name)[1].lower()

            # Write to temp file for loaders that need a file path
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                if file_extension == ".pdf":
                    docs = _load_pdf(tmp_path, file_name)
                elif file_extension == ".docx":
                    docs = _load_docx(tmp_path, file_name)
                elif file_extension == ".txt":
                    docs = _load_txt(tmp_path, file_name)
                else:
                    logger.warning(f"Unsupported file type: {file_extension} for {file_name}")
                    continue

                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} pages/sections from '{file_name}'")

            finally:
                # Clean up temp file
                os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"Error loading '{uploaded_file.name}': {e}")
            continue

    return documents


def load_documents_from_directory(directory_path):
    """
    Load all supported documents from a directory (for pre-seeded docs).

    Args:
        directory_path: Path to directory containing documents

    Returns:
        List of LangChain Document objects
    """
    documents = []

    if not os.path.exists(directory_path):
        logger.warning(f"Directory not found: {directory_path}")
        return documents

    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if not os.path.isfile(file_path):
            continue

        file_extension = os.path.splitext(file_name)[1].lower()

        try:
            if file_extension == ".pdf":
                docs = _load_pdf(file_path, file_name)
            elif file_extension == ".docx":
                docs = _load_docx(file_path, file_name)
            elif file_extension == ".txt":
                docs = _load_txt(file_path, file_name)
            else:
                continue

            documents.extend(docs)
            logger.info(f"Loaded {len(docs)} pages/sections from '{file_name}'")

        except Exception as e:
            logger.error(f"Error loading '{file_name}' from directory: {e}")
            continue

    return documents


def chunk_documents(documents):
    """
    Split documents into chunks for embedding.

    Args:
        documents: List of LangChain Document objects

    Returns:
        List of chunked Document objects with metadata
    """
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        chunks = splitter.split_documents(documents)

        # Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks

    except Exception as e:
        logger.error(f"Error chunking documents: {e}")
        raise


# =============================================================================
# PRIVATE LOADER HELPERS
# =============================================================================

def _load_pdf(file_path, source_name):
    """Load a PDF file."""
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    documents = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            documents.append(Document(
                page_content=text,
                metadata={"source": source_name, "page": i + 1, "type": "pdf"}
            ))

    return documents


def _load_docx(file_path, source_name):
    """Load a DOCX file."""
    import docx2txt

    text = docx2txt.process(file_path)
    if text and text.strip():
        return [Document(
            page_content=text,
            metadata={"source": source_name, "page": 1, "type": "docx"}
        )]
    return []


def _load_txt(file_path, source_name):
    """Load a TXT file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    if text and text.strip():
        return [Document(
            page_content=text,
            metadata={"source": source_name, "page": 1, "type": "txt"}
        )]
    return []
