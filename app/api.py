import hashlib
import logging
import os
from pathlib import Path
from threading import Lock
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import LLM_MODEL, RAW_PDF_DIR
from rag_pipeline import DocumentAssistant


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

APP_NAME = "DocumentAssistant API"
APP_VERSION = "2.0.0"

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

ALLOWED_EXTENSION = ".pdf"

API_KEY_ENV_NAME = "DOCUMENTASSISTANT_API_KEY"


# ============================================================
# 2. LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("documentassistant.api")


# ============================================================
# 3. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=APP_NAME,
    description=(
        "Upload PDFs and ask questions using "
        "hybrid retrieval-augmented generation."
    ),
    version=APP_VERSION,
)


# ============================================================
# 4. APPLICATION STATE
# ============================================================

# The currently loaded assistant.
assistant: DocumentAssistant | None = None

# Metadata about the currently loaded document.
loaded_document: dict | None = None

# Protect shared assistant state during uploads and queries.
state_lock = Lock()


# ============================================================
# 5. PYDANTIC MODELS
# ============================================================

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question to ask about the uploaded PDF.",
    )


class SourceResponse(BaseModel):
    source: str
    page: int | str | None = None
    chunk_id: int | str | None = None
    citation: str | None = None
    content: str | None = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceResponse]


class UploadResponse(BaseModel):
    message: str
    filename: str
    document_id: str
    chunks_created: int


class HealthResponse(BaseModel):
    status: str
    application: str
    version: str
    assistant_loaded: bool
    llm_model: str


# ============================================================
# 6. API-KEY AUTHENTICATION
# ============================================================

def verify_api_key(
    x_api_key: str | None = Header(
        default=None,
        alias="X-API-Key",
    ),
) -> None:
    """
    Validate the API key supplied in the X-API-Key header.
    """

    expected_api_key = os.getenv(API_KEY_ENV_NAME)

    # Fail closed if the server has not been configured
    # with an API key.
    if not expected_api_key:
        logger.error("API key is not configured.")

        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured.",
        )

    if not x_api_key or x_api_key != expected_api_key:
        logger.warning("Request rejected: invalid or missing API key.")

        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
        )


# ============================================================
# 7. HEALTH ENDPOINT
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
def health_check() -> HealthResponse:
    """
    Check whether the API is running.
    """

    return HealthResponse(
        status="ok",
        application=APP_NAME,
        version=APP_VERSION,
        assistant_loaded=assistant is not None,
        llm_model=LLM_MODEL,
    )


# ============================================================
# 8. PDF UPLOAD ENDPOINT
# ============================================================

@app.post(
    "/upload",
    response_model=UploadResponse,
    tags=["Documents"],
    dependencies=[Depends(verify_api_key)],
)
def upload_pdf(
    file: UploadFile = File(...),
) -> UploadResponse:
    """
    Upload a PDF and initialise the RAG assistant.
    """

    global assistant, loaded_document

    original_filename = Path(file.filename or "").name

    if not original_filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

    if Path(original_filename).suffix.lower() != ALLOWED_EXTENSION:
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    # Read at most MAX_FILE_SIZE + 1 bytes so oversized
    # uploads can be rejected without reading the whole file.
    pdf_bytes = file.file.read(MAX_FILE_SIZE + 1)

    if not pdf_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(pdf_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="PDF exceeds the 20 MB upload limit.",
        )

    # Check the PDF signature. This is a basic validation,
    # not a complete guarantee that the file is a valid PDF.
    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file does not have a valid PDF signature.",
        )

    RAW_PDF_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    document_id = uuid4().hex

    # Use a generated filename rather than trusting the
    # user's filename for the physical storage path.
    stored_filename = f"{document_id}.pdf"

    pdf_path = RAW_PDF_DIR / stored_filename

    try:
        pdf_path.write_bytes(pdf_bytes)

        logger.info(
            "PDF upload received: filename=%s document_id=%s size=%s",
            original_filename,
            document_id,
            len(pdf_bytes),
        )

        # Build the new assistant before replacing the current one.
        # If processing fails, the previous assistant remains available.
        new_assistant = DocumentAssistant()

        new_assistant.load_pdf(pdf_path)

        if not new_assistant.documents:
            raise ValueError(
                "No usable text chunks were extracted from the PDF."
            )

        document_info = {
            "document_id": document_id,
            "filename": original_filename,
            "stored_filename": stored_filename,
            "sha256": hashlib.sha256(pdf_bytes).hexdigest(),
        }

        # Swap the active assistant and its metadata together.
        with state_lock:
            assistant = new_assistant
            loaded_document = document_info

        logger.info(
            "PDF processed successfully: document_id=%s chunks=%s",
            document_id,
            len(new_assistant.documents),
        )

        return UploadResponse(
            message="PDF uploaded and processed successfully.",
            filename=original_filename,
            document_id=document_id,
            chunks_created=len(new_assistant.documents),
        )

    except HTTPException:
        pdf_path.unlink(missing_ok=True)
        raise

    except Exception as error:
        # Avoid returning internal stack traces or system details
        # to API clients.
        pdf_path.unlink(missing_ok=True)

        logger.exception(
            "PDF processing failed for document_id=%s",
            document_id,
        )

        raise HTTPException(
            status_code=500,
            detail="PDF processing failed. Check the server logs.",
        ) from error

    finally:
        file.file.close()


# ============================================================
# 9. QUESTION-ANSWERING ENDPOINT
# ============================================================

@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["Questions"],
    dependencies=[Depends(verify_api_key)],
)
def query_document(
    request: QueryRequest,
) -> QueryResponse:
    """
    Answer a question using the currently loaded PDF.
    """

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot contain only whitespace.",
        )

    if len(question) > 2000:
        raise HTTPException(
            status_code=422,
            detail="Question cannot exceed 2000 characters.",
        )

    with state_lock:
        current_assistant = assistant

        if current_assistant is None:
            raise HTTPException(
                status_code=409,
                detail="No PDF is loaded. Upload a PDF first.",
            )

        try:
            result = current_assistant.ask(question)

        except ValueError as error:
            logger.warning(
                "Invalid query received: %s",
                error,
            )

            raise HTTPException(
                status_code=422,
                detail="The question could not be processed.",
            ) from error

        except Exception as error:
            logger.exception("Question processing failed.")

            raise HTTPException(
                status_code=500,
                detail="Question processing failed. Check the server logs.",
            ) from error

    # Convert the dictionary returned by DocumentAssistant
    # into a validated Pydantic response.
    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        sources=result["sources"],
    )


# ============================================================
# 10. GENERAL ERROR HANDLER
# ============================================================

@app.exception_handler(Exception)
async def unexpected_error_handler(request, error: Exception):
    """
    Return a generic response for unexpected server errors.
    """

    logger.exception(
        "Unhandled exception on %s %s",
        request.method,
        request.url.path,
        exc_info=error,
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected server error occurred."
        },
    )