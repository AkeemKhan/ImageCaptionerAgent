"""Command-line entry point: start the web app."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import uvicorn
from dotenv import find_dotenv, load_dotenv

from .app import create_app
from .settings import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Image Captioner Agent web app")
    parser.add_argument("--settings", type=Path, help="path to settings.yaml")
    parser.add_argument("--host", help="override the bind address")
    parser.add_argument("--port", type=int, help="override the port")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # .env in the working directory or any parent; real environment variables win.
    load_dotenv(find_dotenv(usecwd=True))
    settings = load_settings(args.settings)
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        logging.getLogger(__name__).warning(
            "ANTHROPIC_API_KEY is not set; chat requests will fail until it is."
        )
    uvicorn.run(
        create_app(settings),
        host=args.host or settings.host,
        port=args.port or settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
