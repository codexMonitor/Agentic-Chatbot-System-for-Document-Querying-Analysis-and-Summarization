from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from doc_agent.api.deps import get_doc_agent
from doc_agent.api.schemas import (
    ChatRequest,
    ChatResponse,
    DeleteDocumentResponse,
    DocumentListResponse,
    HealthResponse,
    UploadDocumentResponse,
)
from doc_agent.app_service import (
    DocAgentApp,
    DocumentTooLargeError,
    UnsupportedDocumentTypeError,
)
from doc_agent.config import Settings, load_settings
from doc_agent.services import Services, build_services


def create_app(services: Services | None = None, settings: Settings | None = None) -> FastAPI:
    if (services is None) != (settings is None):
        raise ValueError("services and settings must be provided together")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        if app.state.doc_agent is None:
            active_settings = load_settings()
            app.state.doc_agent = DocAgentApp(build_services(active_settings), active_settings)
        try:
            yield
        finally:
            app.state.doc_agent.close()

    app = FastAPI(lifespan=lifespan)
    app.state.doc_agent = (
        DocAgentApp(services, settings) if services is not None and settings is not None else None
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/documents", response_model=UploadDocumentResponse)
    async def upload_document(
        file: Annotated[UploadFile, File()],
        service: Annotated[DocAgentApp, Depends(get_doc_agent)],
    ) -> dict[str, str | int]:
        content = await file.read(service.max_upload_bytes + 1)
        try:
            return service.upload_document(file.filename or "", content)
        except UnsupportedDocumentTypeError as error:
            raise HTTPException(status_code=415, detail=str(error)) from error
        except DocumentTooLargeError as error:
            raise HTTPException(status_code=413, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        finally:
            await file.close()

    @app.get("/documents", response_model=DocumentListResponse)
    def list_documents(
        service: Annotated[DocAgentApp, Depends(get_doc_agent)],
    ) -> dict[str, list[str]]:
        return {"documents": service.list_documents()}

    @app.delete("/documents/{source}", response_model=DeleteDocumentResponse)
    def delete_document(
        source: str, service: Annotated[DocAgentApp, Depends(get_doc_agent)]
    ) -> dict[str, int]:
        return {"removed": service.delete_document(source)}

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest, service: Annotated[DocAgentApp, Depends(get_doc_agent)]) -> dict:
        return service.chat(request.thread_id, request.message)

    return app


app = create_app()
