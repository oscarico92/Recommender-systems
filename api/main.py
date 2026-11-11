"""ScoutAI API. Run with: uvicorn api.main:app --reload

Minimal app so that /log_interaction can be tested now. The /recommend
endpoint is added by the Systems Engineer.
"""

from fastapi import FastAPI

from api.interactions import router as interactions_router

app = FastAPI(title="ScoutAI API")
app.include_router(interactions_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
