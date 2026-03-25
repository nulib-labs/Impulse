from dataclasses import dataclass
from enum import Enum


@dataclass
class WorkMetadata:
    """Class for capturing metadata about a work"""
    creators: list | None
    title: str | None
    medium: str | None
    language: str | None

@dataclass
class PageProcessingMeta:
    deskew_angle: float | None
    noise_level: str

@dataclass
class Page:
    _id: str
    work_id: str
    page_number: int
    status: str
    representations: dict
    processed: dict


@dataclass
class Work:
    """Class for tracking work metadata"""
    _id: str
    page_count: int
    metadata: dict
    status: str = "PENDING"
