import os
import subprocess

print(os.listdir())
for dir in os.listdir("lost_flows"):
    accession_number = dir
    _ = subprocess.run(
        [
            "uv",
            "run",
            "impulse/main.py",
            "--input",
            f"lost_flows/{dir}",
            "--accession_number",
            str(accession_number),
        ],
        check=True,
    )
