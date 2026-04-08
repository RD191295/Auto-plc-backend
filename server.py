"""FastAPI server to expose the LangGraph multi-agent pipeline."""

import os
import json
import time
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

from graph import agent_graph
from state import GraphState, ProjectFile

app = FastAPI(title="ControlForge Multi-Agent Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    instruction: str
    platform: str = "codesys"
    ai_model: str = "fast"
    project_id: str = ""
    user_id: str = ""
    existing_files: List[Dict] = []
    conversation_history: List[Dict] = []


class GenerateResponse(BaseModel):
    run_id: str
    project_name: str
    agent_messages: List[Dict]
    files: List[Dict]
    quality_score: Optional[float] = None
    approved: bool = False


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Run the multi-agent pipeline to generate a PLC project."""
    supabase = None
    try:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if url and key:
            supabase = create_client(url, key)
    except:
        pass

    # Create run record
    run_id = ""
    if supabase and req.user_id and req.project_id:
        try:
            result = supabase.table("agent_runs").insert({
                "project_id": req.project_id,
                "user_id": req.user_id,
                "instruction": req.instruction,
                "status": "running",
            }).execute()
            run_id = result.data[0]["id"]

            # Pre-create step records for real-time tracking
            agents = ["Requirement Agent", "Planning Agent", "Retrieval Agent",
                      "Coding Agent", "Debugging Agent", "Verification Agent"]
            for i, agent_name in enumerate(agents):
                supabase.table("agent_steps").insert({
                    "run_id": run_id,
                    "agent_name": agent_name,
                    "step_order": i,
                    "status": "pending",
                }).execute()
        except Exception as e:
            print(f"Failed to create run: {e}")

    # Build initial state
    existing = [ProjectFile(**f) for f in req.existing_files] if req.existing_files else []

    initial_state = GraphState(
        instruction=req.instruction,
        platform=req.platform,
        ai_model=req.ai_model,
        project_id=req.project_id,
        user_id=req.user_id,
        run_id=run_id,
        existing_files=existing,
        conversation_history=req.conversation_history,
    )

    # Run the graph
    try:
        result = agent_graph.invoke(initial_state.model_dump())

        # Update run as completed
        if supabase and run_id:
            supabase.table("agent_runs").update({
                "status": "completed",
                "current_agent": None,
                "result": {
                    "files": [f.model_dump() if hasattr(f, 'model_dump') else f for f in result.get("generated_files", [])],
                    "agent_messages": [m.model_dump() if hasattr(m, 'model_dump') else m for m in result.get("agent_messages", [])],
                },
            }).eq("id", run_id).execute()

        return GenerateResponse(
            run_id=run_id,
            project_name=result.get("plan", {}).get("project_name", "Generated Project") if isinstance(result.get("plan"), dict) else (result.get("plan").project_name if result.get("plan") else "Generated Project"),
            agent_messages=[m.model_dump() if hasattr(m, 'model_dump') else m for m in result.get("agent_messages", [])],
            files=[f.model_dump() if hasattr(f, 'model_dump') else f for f in result.get("generated_files", [])],
            quality_score=result.get("quality_score"),
            approved=result.get("approved", False),
        )

    except Exception as e:
        if supabase and run_id:
            supabase.table("agent_runs").update({
                "status": "error",
                "error": str(e),
            }).eq("id", run_id).execute()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "controlforge-multi-agent"}


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
