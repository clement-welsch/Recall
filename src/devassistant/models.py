from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4


@dataclass
class KnowledgeDomain:
    """Identify a named knowledge collection and its document directory."""

    name: str
    directory: Path
    description: str = ""
    id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Name must be a non-empty string")

        if not isinstance(self.directory, (str, Path)) or not str(
            self.directory
        ).strip():
            raise ValueError("Directory must be a non-empty path")
        self.directory = Path(self.directory)

        if not isinstance(self.description, str):
            raise ValueError("Description must be a string")

        if not isinstance(self.id, UUID):
            raise ValueError("ID must be a UUID")
