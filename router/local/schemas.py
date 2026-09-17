from pydantic import BaseModel, Field


class OllamaModel(BaseModel):
    name: str
    role: str
    capabilities: list[str] = Field(default_factory=list)
    size: int | None = None
    family: str | None = None


class ModelsResponse(BaseModel):
    embedding_model: str
    generate_models: list[OllamaModel]
    embedding_models: list[OllamaModel]
    models: list[OllamaModel]


class EmbedRequest(BaseModel):
    texts: list[str]
    as_query: bool = False


class EmbedResponse(BaseModel):
    model: str
    dimensions: int
    count: int
    embeddings: list[list[float]]


class ReferenceIn(BaseModel):
    texts: list[str]
    filename: str | None = None


class ReferenceOut(BaseModel):
    ids: list[str]
    count: int
    embedding_model: str


class RunRequest(BaseModel):
    instruction: str
    generate_model: str | None = None
    target_text: str | None = None
    match_count: int = 5


class RunResponse(BaseModel):
    message: str
    generate_model: str
    embedding_model: str
    target_stored: bool = False
