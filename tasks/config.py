import os

# Impulse configuration

DEBUG = True # Defines which environment variables to use. Should not be used in prod

if DEBUG:
    MONGO_URI = os.getenv("IMPULSE_MONGODB_URI_DEBUG")
else:
    MONGO_URI = os.getenv("IMPULSE_MONGODB_URI")
