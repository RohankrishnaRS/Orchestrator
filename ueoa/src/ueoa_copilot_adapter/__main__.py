"""python -m ueoa_copilot_adapter  -> starts the Teams / Copilot adapter."""

from dotenv import load_dotenv

from ueoa_copilot_adapter.server import main

load_dotenv()
main()