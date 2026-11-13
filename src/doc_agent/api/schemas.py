from pydantic import BaseModel, Field, field_validator


class UploadDocumentResponse(BaseModel):
    source: str
    chunks: int


class DocumentListResponse(BaseModel):
    documents: list[str]


class DeleteDocumentResponse(BaseModel):
    removed: int


class ChatRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    message: str = Field(min_length=1)

    @field_validator("thread_id", "message")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value


class SourceReference(BaseModel):
    n: int
    source: str
    position: int


class ChatResponse(BaseModel):
    intent: str
    answer: str
    sources: list[SourceReference]


class HealthResponse(BaseModel):
    status: str
