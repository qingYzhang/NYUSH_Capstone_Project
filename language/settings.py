import os
from dotenv import load_dotenv


# Load .env file
load_dotenv()

# Assign env variables
OPENAI_API_KEY_GPT4 = os.getenv("OPENAI_API_KEY_GPT4")
OPENAI_API_KEY_BURST_GPT4 = os.getenv("OPENAI_API_KEY_BURST_GPT4")
