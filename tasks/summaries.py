import re
from tasks.helpers import parse_s3_path, get_s3_content, _get_db
import base64
import fitz
from fireworks.core.firework import FWAction, FireTaskBase
import os

class SummariesTask(FireTaskBase):
    _fw_name = "Summaries Task"

    @staticmethod
    def parse_s3_path(s3_path: str) -> tuple[str, str]:
        path = re.sub(r"^s3a?://", "", s3_path)
        parts = path.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket, key

    @staticmethod
    def pdf_to_images(pdf_bytes: bytes) -> list[str]:
        """Convert each PDF page to a base64-encoded PNG string."""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        images = []
        for page in doc:
            pix = page.get_pixmap(dpi=150)
            img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
            images.append(img_b64)
        return images

    @staticmethod
    def image_to_b64(image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("utf-8")

    @staticmethod
    def ask_ai(document_text=None, pdf_bytes=None, image_bytes=None):
        from ollama import chat

        prompt = """
        Provide a short summary of the document.
        Do not return anything except plain text summary.
        Use markdown only if necessary.
        """
        message = {
            "role": "user",
            "content": prompt,
        }

        if pdf_bytes:
            # Convert PDF pages to images — gemma3 can't read raw PDF bytes
            message["images"] = SummariesTask.pdf_to_images(pdf_bytes)

        elif image_bytes:
            message["images"] = [SummariesTask.image_to_b64(image_bytes)]

        if document_text:
            message["content"] += f"\n\n{document_text}"

        response = chat(
            model="gemma3:27b",
            messages=[message],
        )
        return response.message.content

    def run_task(self, fw_spec):
        document = fw_spec["document"]

        IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

        # Case 1: S3 or local file path
        if isinstance(document, str) and (
            document.startswith("s3://")
            or document.startswith("s3a://")
            or "/" in document
        ):
            ext = os.path.splitext(document)[1].lower()

            if document.startswith(("s3://", "s3a://")):
                print("Document starts with s3, pulling from bucket")
                content_bytes = get_s3_content(document)
            else:
                with open(document, "rb") as f:
                    content_bytes = f.read()

            if ext == ".pdf":
                summary = self.ask_ai(pdf_bytes=content_bytes)
            elif ext in IMAGE_EXTENSIONS:
                summary = self.ask_ai(image_bytes=content_bytes)
            else:
                # Treat as text file
                summary = self.ask_ai(document_text=content_bytes.decode("utf-8"))

        # Case 2: List of S3/file paths
        elif isinstance(document, list):
            document_text = []
            for path in document:
                content = self.get_s3_content(path)
                document_text.append(content.decode("utf-8"))
            summary = self.ask_ai(document_text="\n".join(document_text))

        # Case 3: Raw string text
        elif isinstance(document, str):
            summary = self.ask_ai(document_text=document)

        else:
            raise ValueError(f"Unsupported document type: {type(document)}")

        print(summary)
        return FWAction(update_spec={"document_summary": summary})
