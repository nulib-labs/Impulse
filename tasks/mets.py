from typing import override
import re
import boto3
from fireworks.core.firework import FWAction, FireTaskBase
from loguru import logger
import os
from tasks.helpers import get_s3_content

class METSXMLToHathiTrustManifestTask(FireTaskBase):
    _fw_name = "MetsXML To HathiTrust Manifest Task"
    path: str

    @staticmethod
    def is_s3_path(path: str) -> bool:
        """
        Check if the path is an S3 URI.
        Supports both s3:// and s3a:// formats.
        """
        return bool(re.match(r"^s3a?://", path))

    @staticmethod
    def parse_s3_path(s3_path: str) -> tuple[str, str]:
        """
        Parse S3 path into bucket and key.

        Args:
            s3_path: S3 URI in format s3://bucket/key or s3a://bucket/key

        Returns:
            Tuple of (bucket, key)
        """
        # Remove s3:// or s3a:// prefix
        path = re.sub(r"^s3a?://", "", s3_path)
        # Split into bucket and key
        parts = path.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket, key

    def save_to_s3(self, s3_path: str, content: str) -> str:
        """
        Save string content to S3.

        Args:
            s3_path: S3 URI (e.g. s3://bucket/key)
            content: File content as a string
        """
        bucket, key = self.parse_s3_path(s3_path)

        s3_client = boto3.client("s3")

        # Convert string to bytes
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=content.encode("utf-8"),  # Encode string as bytes
        )

        return s3_path

    def _convert_mets_to_yml(*args):
        """
        FireWorks task function: Convert a METS XML file from S3 into a YAML file
        following HathiTrust ingest specs, and save the YAML back to S3.
        args[0] = s3_key of the METS XML file in S3
        args[1] = filename
        args[2] = accession_number
        """
        from lxml import etree as ET

        s3_key = args[0]
        filename = args[1]
        accession_number = args[2]

        s3_impulse = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("IMPULSE_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("IMPULSE_SECRET_ACCESS_KEY"),
        )

        output_filename = filename.replace(".xml", ".yaml")
        logger.info(f"Value of output filename: {output_filename}")
        response = s3_impulse.get_object(Bucket="nu-impulse-production", Key=s3_key)
        data = response["Body"].read()

        # === Step 2: Parse the XML directly from memory ===
        try:
            parser = ET.XMLParser(remove_blank_text=True)
            tree = ET.ElementTree(ET.fromstring(data, parser))
            root = tree.getroot()
        except ET.XMLSyntaxError as e:
            print(f"Error: Invalid XML file: {e}")
            raise RuntimeError("Invalid METS XML file") from e

        ns = {
            "xmlns": "http://www.loc.gov/METS/",
            "xlink": "http://www.w3.org/1999/xlink",
        }

        def find_filename_by_file_id(file_id):
            node = root.xpath(
                f"//xmlns:file[@ID='{file_id}']/xmlns:FLocat", namespaces=ns
            )
            if not node:
                return None
            href = node[0].get("{http://www.w3.org/1999/xlink}href", "")
            return href[7:] if href.startswith("file://") else href

        # === Step 3: Extract header information ===
        mets_hdr = root.xpath("//xmlns:metsHdr", namespaces=ns)
        capture_date = mets_hdr[0].get("CREATEDATE") + "-06:00" if mets_hdr else None

        suprascan = False
        scanning_order_rtl = False
        reading_order_rtl = False
        resolution = 400

        yaml_lines = []
        yaml_lines.append(f"capture_date: {capture_date}")
        if suprascan:
            yaml_lines.append("scanner_make: SupraScan")
            yaml_lines.append("scanner_model: Quartz A1")
        else:
            yaml_lines.append("scanner_make: Kirtas")
            yaml_lines.append("scanner_model: APT 1200")
        yaml_lines.append(
            'scanner_user: "Northwestern University Library: Repository & Digital Curation"'
        )
        yaml_lines.append(f"contone_resolution_dpi: {resolution}")
        yaml_lines.append(f"image_compression_date: {capture_date}")
        yaml_lines.append("image_compression_agent: northwestern")
        yaml_lines.append('image_compression_tool: ["LIMB v4.5.0.0"]')
        yaml_lines.append(
            f"scanning_order: {'right-to-left' if scanning_order_rtl else 'left-to-right'}"
        )
        yaml_lines.append(
            f"reading_order: {'right-to-left' if reading_order_rtl else 'left-to-right'}"
        )
        yaml_lines.append("pagedata:")

        # === Step 4: Logical structMap page iteration ===
        logical_pages = root.xpath(
            '//xmlns:structMap[@TYPE="logical"]//xmlns:div[@TYPE="page"]', namespaces=ns
        )

        for element in logical_pages:
            fileptr = element.xpath(
                "./xmlns:fptr[starts-with(@FILEID, 'JP2')]", namespaces=ns
            )
            if not fileptr:
                continue
            file_id = fileptr[0].get("FILEID")
            page_filename = find_filename_by_file_id(file_id)
            if not page_filename:
                continue

            parent = element.getparent()
            parent_label = parent.get("LABEL", "")
            parent_type = parent.get("TYPE", "")
            orderlabel = element.get("ORDERLABEL", "")
            line = None

            # Label logic
            if element == parent[0]:
                if (
                    parent_label == "Cover"
                    and parent_type == "cover"
                    and parent == logical_pages[0].getparent()
                ):
                    label = "FRONT_COVER"
                    line = f'{page_filename}: {{ label: "{label}" }}'
                elif parent_label == "Front Matter" and orderlabel:
                    line = f'{page_filename}: {{ orderlabel: "{orderlabel}" }}'
                elif parent_label == "Title":
                    label = "TITLE"
                    line = (
                        f'{page_filename}: {{ orderlabel: "{orderlabel}", label: "{label}" }}'
                        if orderlabel
                        else f'{page_filename}: {{ label: "{label}" }}'
                    )
                elif parent_label == "Contents":
                    label = "TABLE_OF_CONTENTS"
                    line = (
                        f'{page_filename}: {{ orderlabel: "{orderlabel}", label: "{label}" }}'
                        if orderlabel
                        else f'{page_filename}: {{ label: "{label}" }}'
                    )
                elif parent_label == "Preface":
                    label = "PREFACE"
                    line = (
                        f'{page_filename}: {{ orderlabel: "{orderlabel}", label: "{label}" }}'
                        if orderlabel
                        else f'{page_filename}: {{ label: "{label}" }}'
                    )
                elif parent_label.startswith("Chapter") or parent_label == "Appendix":
                    label = "CHAPTER_START"
                    line = (
                        f'{page_filename}: {{ orderlabel: "{orderlabel}", label: "{label}" }}'
                        if orderlabel
                        else f'{page_filename}: {{ label: "{label}" }}'
                    )
                elif parent_label in ("Notes", "Bibliography"):
                    label = "REFERENCES"
                    line = (
                        f'{page_filename}: {{ orderlabel: "{orderlabel}", label: "{label}" }}'
                        if orderlabel
                        else f'{page_filename}: {{ label: "{label}" }}'
                    )
                elif parent_label == "Index":
                    label = "INDEX"
                    line = (
                        f'{page_filename}: {{ orderlabel: "{orderlabel}", label: "{label}" }}'
                        if orderlabel
                        else f'{page_filename}: {{ label: "{label}" }}'
                    )
                elif parent_label == "Cover" and parent_type == "cover":
                    label = "BACK_COVER"
                    line = f'{page_filename}: {{ label: "{label}" }}'
            else:
                if orderlabel:
                    line = f'{page_filename}: {{ orderlabel: "{orderlabel}" }}'

            if line:
                yaml_lines.append("    " + line)

        # === Step 5: Write YAML to S3 ===
        yaml_content: str = "\n".join(yaml_lines)
        return yaml_content

    @override
    def run_task(self, fw_spec: dict[str, str]) -> FWAction:
        logger.info("Now loading content from S3")
        input_path = fw_spec["input_path"]
        output_path = fw_spec["output_path"]
        content = get_s3_content(input_path)

        yaml_content: str = self.convert_mets_to_yml(content)
        s3_path = self.save_to_s3(output_path, yaml_content)

        return FWAction(update_spec={"hathitrust_yaml_path": s3_path})


