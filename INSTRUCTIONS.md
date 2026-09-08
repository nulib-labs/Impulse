# Instructions for Submitting to Your Impulse Instance

Set up an aws profile with s3 full access called "impulse"
Set up an environment variable called IMPULSE_MONGODB_URI that is a link for your username and mongodb password string

Submit jobs from a samba share:

From the Impulse git repo root (Windows):

Source your env vars:

`source .env`

To download
`.venv/Scripts/python.exe -m cli download IMPULSE_IDENTIFIER --out PATH_TO_OUT_FOLDER`

From the Impulse git repo root (Windows):

To upload
`.venv/Scripts/python.exe -m cli upload IMPULSE_IDENTIFIER --from PATH_TO_INPUT_FOLDER`

