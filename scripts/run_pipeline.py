from dataclasses import dataclass
from PIL import Image
from argparse import ArgumentParser

parser = ArgumentParser(
    prog="Impulse",
    description="Run jobs locally"
)

@dataclass
class ImpulseItem:
    """ Class for passing image meta/data to impulse."""
    impulse_identifier: str
    page_number: int
@dataclass
class ImpulseInputItem(ImpulseItem):
    image_data: Image.Image

@dataclass
class ImpulseOutputItem(ImpulseItem):
    layout_data: dict
    ocr_data: dict
    extraction_model: str = "surya-2"


