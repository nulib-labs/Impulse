import os
import subprocess

print("=" * 50)
print(
"""
 mmmmm  m    m mmmmm  m    m m       mmmm  mmmmmm
   #    ##  ## #   "# #    # #      #"   " #     
   #    # ## # #mmm#" #    # #      "#mmm  #mmmmm
   #    # "" # #      #    # #          "# #     
 mm#mm  #    # #      "mmmm" #mmmmm "mmm#" #mmmmm
""")
print("=" * 50)
uri = input("Enter your MongoDB URI: ")

# Persist to .bashrc for future sessions
bashrc_path = os.path.expanduser("~/.bashrc")
with open(bashrc_path, "a") as f:
    f.write(f"\nexport IMPULSE_MONGODB_URI='{uri}'\n")

# Set it in the current process for immediate use
os.environ['IMPULSE_MONGODB_URI'] = uri

print("Saved! For new terminal sessions, run: source ~/.bashrc")
