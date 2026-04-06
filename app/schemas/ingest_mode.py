from enum import Enum


class IngestMode(str, Enum):
    LEGACY = "legacy"
    SEMANTIC = "semantic"