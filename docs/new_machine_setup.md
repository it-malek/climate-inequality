# Historical local setup

Commands and reported values here belong to this historical revision.
Personal machine-recovery and session instructions have been omitted.

## Fresh clone

```bash
git clone https://github.com/it-malek/climate-inequality.git
cd climate-inequality
uv sync --extra dev
uv run pytest -q
uv run ruff check src tests app
uv run streamlit run app/streamlit_app.py
```

The dashboard reads the committed bundle. Local data rebuilding is separate.
