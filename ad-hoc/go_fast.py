import os
import subprocess

print(os.listdir())

for dir in os.listdir("Year2"):
    accession_number = dir

    _ = subprocess.run(
        [
            "uv",
            "run",
            "main.py",
            "--input",
            f"./Year2/{dir}",
            "--accession_number",
            str(accession_number),
        ],
        check=True,
    )
