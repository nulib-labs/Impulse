"""METS XML to HathiTrust YAML manifest conversion."""

from __future__ import annotations

from loguru import logger


def convert_mets_to_yaml(xml_bytes: bytes) -> str:
    """Convert a METS XML document to a HathiTrust ingest YAML string.

    Args:
        xml_bytes: Raw bytes of the METS XML file.

    Returns:
        YAML content as a string.
    """
    from lxml import etree as ET

    try:
        parser = ET.XMLParser(remove_blank_text=True)
        tree = ET.ElementTree(ET.fromstring(xml_bytes, parser))
        root = tree.getroot()
    except ET.XMLSyntaxError as e:
        logger.error(f"Invalid METS XML: {e}")
        raise RuntimeError("Invalid METS XML file") from e

    ns = {
        "xmlns": "http://www.loc.gov/METS/",
        "xlink": "http://www.w3.org/1999/xlink",
    }

    def find_filename_by_file_id(file_id: str) -> str | None:
        node = root.xpath(f"//xmlns:file[@ID='{file_id}']/xmlns:FLocat", namespaces=ns)
        if not node:
            return None
        href = node[0].get("{http://www.w3.org/1999/xlink}href", "")
        return href[7:] if href.startswith("file://") else href

    # ── Header info ─────────────────────────────────────────────────────

    mets_hdr = root.xpath("//xmlns:metsHdr", namespaces=ns)
    capture_date = mets_hdr[0].get("CREATEDATE") + "-06:00" if mets_hdr else None

    suprascan = False
    scanning_order_rtl = False
    reading_order_rtl = False
    resolution = 400

    yaml_lines: list[str] = []
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

    # ── Logical structMap page iteration ────────────────────────────────

    logical_pages = root.xpath(
        '//xmlns:structMap[@TYPE="logical"]//xmlns:div[@TYPE="page"]',
        namespaces=ns,
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
        line: str | None = None

        if element == parent[0]:
            if (
                parent_label == "Cover"
                and parent_type == "cover"
                and parent == logical_pages[0].getparent()
            ):
                line = f'{page_filename}: {{ label: "FRONT_COVER" }}'
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
                line = f'{page_filename}: {{ label: "BACK_COVER" }}'
        else:
            if orderlabel:
                line = f'{page_filename}: {{ orderlabel: "{orderlabel}" }}'

        if line:
            yaml_lines.append("    " + line)

    return "\n".join(yaml_lines)
