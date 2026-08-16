"""Static file server for debugging/exploration pages — plain HTML/JS
files, no build step. Each page under pages/ is a self-contained
exploration; add new ones by dropping a new .html file there and linking
it from pages/index.html.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

PAGES_DIR = Path(__file__).parent / "pages"

app = FastAPI()
app.mount("/", StaticFiles(directory=PAGES_DIR, html=True), name="pages")
