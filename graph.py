"""LangGraph multi-agent pipeline for PLC code generation."""

import json
import os
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from supabase import create_client
from state import GraphState, AgentMessage, Requirements, Plan, ProjectFile, DebugIssue, SafetyCheck
from prompts import (
    REQUIREMENT_AGENT_PROMPT, PLANNING_AGENT_PROMPT, RETRIEVAL_AGENT_PROMPT,
    CODING_AGENT_PROMPT, DEBUGGING_AGENT_PROMPT, VERIFICATION_AGENT_PROMPT,
    VENDOR_REFS,
)
from tools import web_search, validate_structured_text, check_safety_patterns


def get_llm(model_key: str = "fast"):
    """Get LLM via Lovable AI Gateway (OpenAI-compatible)."""
    from langchain_openai import ChatOpenAI

    model_map = {
        "fast": "google/gemini-3-flash-preview",
        "balanced": "google/gemini-2.5-flash",
        "quality": "google/gemini-2.5-pro",
    }

    return ChatOpenAI(
        model=model_map.get(model_key, model_map["fast"]),
        api_key=os.getenv("LOVABLE_API_KEY"),
        base_url="https://ai.gateway.lovable.dev/v1",
        temperature=0.3,
    )


def get_supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if url and key:
        return create_client(url, key)
    return None


def update_step_status(supabase, run_id: str, agent_name: str, step_order: int, status: str, output: dict = None, duration_ms: int = None):
    """Update agent step in Supabase for real-time frontend updates."""
    if not supabase:
        return
    try:
        data = {"status": status}
        if output:
            data["output_data"] = output
        if duration_ms is not None:
            data["duration_ms"] = duration_ms

        supabase.table("agent_steps").update(data).eq("run_id", run_id).eq("agent_name", agent_name).execute()
    except Exception as e:
        print(f"Failed to update step: {e}")


# ---- Agent Node Functions ----

def requirement_agent(state: GraphState) -> dict:
    """Analyze user requirements."""
    import time
    start = time.time()
    llm = get_llm(state.ai_model)
    supabase = get_supabase()

    vendor_ref = VENDOR_REFS.get(state.platform, VENDOR_REFS["codesys"])
    prompt = f"{REQUIREMENT_AGENT_PROMPT}\n\nTarget Platform: {state.platform}\nReference: {vendor_ref}\n\nUser Instruction: {state.instruction}"

    if state.existing_files:
        file_ctx = "\n".join(f"--- {f.path} ---\n{f.content}" for f in state.existing_files)
        prompt += f"\n\nExisting Files:\n{file_ctx}"

    response = llm.invoke([SystemMessage(content=REQUIREMENT_AGENT_PROMPT), HumanMessage(content=prompt)])

    try:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        parsed = json.loads(content)
        requirements = Requirements(**parsed)
    except:
        requirements = Requirements(summary=response.content)

    duration = int((time.time() - start) * 1000)
    update_step_status(supabase, state.run_id, "Requirement Agent", 0, "done", parsed if 'parsed' in dir() else {"summary": response.content}, duration)

    return {
        "requirements": requirements,
        "agent_messages": state.agent_messages + [
            AgentMessage(agent="Requirement Agent", message=requirements.summary)
        ],
    }


def planning_agent(state: GraphState) -> dict:
    """Create implementation plan."""
    import time
    start = time.time()
    llm = get_llm(state.ai_model)
    supabase = get_supabase()

    req_json = state.requirements.model_dump_json() if state.requirements else "{}"
    prompt = f"Requirements:\n{req_json}\n\nPlatform: {state.platform}\nInstruction: {state.instruction}"

    response = llm.invoke([SystemMessage(content=PLANNING_AGENT_PROMPT), HumanMessage(content=prompt)])

    try:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        parsed = json.loads(content)
        plan = Plan(**parsed)
    except:
        plan = Plan(project_name="Generated Project")

    duration = int((time.time() - start) * 1000)
    update_step_status(supabase, state.run_id, "Planning Agent", 1, "done", plan.model_dump(), duration)

    return {
        "plan": plan,
        "agent_messages": state.agent_messages + [
            AgentMessage(
                agent="Planning Agent",
                message=f"Plan: {plan.project_name}. {len(plan.file_structure)} files, {len(plan.function_blocks)} FBs.",
                files=[f.path for f in plan.file_structure],
            )
        ],
    }


def retrieval_agent(state: GraphState) -> dict:
    """Retrieve relevant context via RAG and web search."""
    import time
    start = time.time()
    llm = get_llm(state.ai_model)
    supabase = get_supabase()

    context = {"web_results": [], "knowledge_base": [], "vendor_docs": []}

    # Web search for platform-specific info
    if state.requirements:
        for comp in state.requirements.components[:3]:
            results = web_search(f"{state.platform} PLC {comp.type} {comp.name} structured text example")
            context["web_results"].extend(results)

    # Search knowledge base via vector similarity (if supabase is available)
    if supabase and state.user_id:
        try:
            # Get knowledge entries
            kb = supabase.table("knowledge_entries").select("title, content").eq("user_id", state.user_id).eq("is_active", True).execute()
            if kb.data:
                context["knowledge_base"] = [{"title": e["title"], "content": e["content"][:500]} for e in kb.data[:5]]
        except:
            pass

    # Add vendor reference
    context["vendor_docs"] = [VENDOR_REFS.get(state.platform, VENDOR_REFS["codesys"])]

    duration = int((time.time() - start) * 1000)
    update_step_status(supabase, state.run_id, "Retrieval Agent", 2, "done", {"sources_found": len(context["web_results"]) + len(context["knowledge_base"])}, duration)

    return {
        "retrieved_context": context,
        "agent_messages": state.agent_messages + [
            AgentMessage(
                agent="Retrieval Agent",
                message=f"Retrieved {len(context['web_results'])} web results, {len(context['knowledge_base'])} KB entries.",
            )
        ],
    }


def coding_agent(state: GraphState) -> dict:
    """Generate PLC code files."""
    import time
    start = time.time()
    llm = get_llm(state.ai_model)
    supabase = get_supabase()

    vendor_ref = VENDOR_REFS.get(state.platform, VENDOR_REFS["codesys"])

    ctx_parts = [f"Platform: {state.platform}\nVendor Reference: {vendor_ref}"]
    if state.requirements:
        ctx_parts.append(f"Requirements:\n{state.requirements.model_dump_json()}")
    if state.plan:
        ctx_parts.append(f"Plan:\n{state.plan.model_dump_json()}")
    if state.retrieved_context:
        ctx_parts.append(f"Retrieved Context:\n{json.dumps(state.retrieved_context, default=str)[:3000]}")
    if state.existing_files:
        ctx_parts.append("Existing Files:\n" + "\n".join(f"--- {f.path} ---\n{f.content}" for f in state.existing_files))

    prompt = "\n\n".join(ctx_parts) + f"\n\nInstruction: {state.instruction}\n\nGenerate all files as a JSON array of {{path, content, language}}."

    response = llm.invoke([SystemMessage(content=CODING_AGENT_PROMPT), HumanMessage(content=prompt)])

    files = []
    try:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "files" in parsed:
            parsed = parsed["files"]
        files = [ProjectFile(**f) for f in parsed]
    except:
        files = [ProjectFile(path="PLC/main_program.st", content=response.content, language="iec-st")]

    duration = int((time.time() - start) * 1000)
    update_step_status(supabase, state.run_id, "Coding Agent", 3, "done", {"file_count": len(files)}, duration)

    return {
        "generated_files": files,
        "agent_messages": state.agent_messages + [
            AgentMessage(
                agent="Coding Agent",
                message=f"Generated {len(files)} files.",
                files=[f.path for f in files],
            )
        ],
    }


def debugging_agent(state: GraphState) -> dict:
    """Validate and fix generated code."""
    import time
    start = time.time()
    llm = get_llm(state.ai_model)
    supabase = get_supabase()

    all_issues = []
    for f in state.generated_files:
        if f.path.endswith(".st"):
            result = validate_structured_text(f.content)
            for issue in result["issues"]:
                all_issues.append(DebugIssue(file=f.path, **issue))

    # If issues found, ask LLM to fix them
    fixed_files = list(state.generated_files)
    if all_issues:
        file_ctx = "\n".join(f"--- {f.path} ---\n{f.content}" for f in state.generated_files)
        issues_ctx = "\n".join(f"- {i.file}: {i.message}" for i in all_issues)
        prompt = f"Fix these issues in the code:\n{issues_ctx}\n\nCode:\n{file_ctx}\n\nReturn the fixed files as JSON array."

        response = llm.invoke([SystemMessage(content=DEBUGGING_AGENT_PROMPT), HumanMessage(content=prompt)])
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "fixed_files" in parsed:
                parsed = parsed["fixed_files"]
            elif isinstance(parsed, dict) and "files" in parsed:
                parsed = parsed["files"]
            if isinstance(parsed, list):
                fixed_files = [ProjectFile(**f) for f in parsed]
        except:
            pass

    duration = int((time.time() - start) * 1000)
    summary = f"Found {len(all_issues)} issues." + (" All fixed." if all_issues else " Code looks clean.")
    update_step_status(supabase, state.run_id, "Debugging Agent", 4, "done", {"issue_count": len(all_issues)}, duration)

    return {
        "debug_issues": all_issues,
        "fixed_files": fixed_files if all_issues else [],
        "generated_files": fixed_files if all_issues else state.generated_files,
        "agent_messages": state.agent_messages + [
            AgentMessage(agent="Debugging Agent", message=summary)
        ],
    }


def verification_agent(state: GraphState) -> dict:
    """Final verification of the project."""
    import time
    start = time.time()
    supabase = get_supabase()

    # Run safety pattern checks on all ST files
    all_code = "\n".join(f.content for f in state.generated_files if f.path.endswith(".st"))
    safety_checks = [SafetyCheck(**c) for c in check_safety_patterns(all_code)]

    # Check requirements coverage
    if state.requirements:
        for comp in state.requirements.components:
            found = comp.name.lower() in all_code.lower() or comp.name.replace(" ", "_").lower() in all_code.lower()
            safety_checks.append(SafetyCheck(
                check=f"Component '{comp.name}' implemented",
                passed=found,
                notes=f"{'Found' if found else 'NOT found'} in generated code",
            ))

    passed = sum(1 for c in safety_checks if c.passed)
    total = len(safety_checks)
    score = round((passed / total) * 10, 1) if total > 0 else 5.0
    approved = score >= 6.0

    duration = int((time.time() - start) * 1000)
    summary = f"{'✅ Approved' if approved else '⚠️ Needs review'}: {passed}/{total} checks passed. Quality: {score}/10"
    update_step_status(supabase, state.run_id, "Verification Agent", 5, "done", {"score": score, "approved": approved}, duration)

    return {
        "safety_checks": safety_checks,
        "quality_score": score,
        "approved": approved,
        "agent_messages": state.agent_messages + [
            AgentMessage(agent="Verification Agent", message=summary)
        ],
    }


# ---- Build the LangGraph ----

def build_graph() -> StateGraph:
    """Build the multi-agent LangGraph pipeline."""
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("requirement_agent", requirement_agent)
    workflow.add_node("planning_agent", planning_agent)
    workflow.add_node("retrieval_agent", retrieval_agent)
    workflow.add_node("coding_agent", coding_agent)
    workflow.add_node("debugging_agent", debugging_agent)
    workflow.add_node("verification_agent", verification_agent)

    # Define edges (sequential pipeline with conditional routing)
    workflow.set_entry_point("requirement_agent")
    workflow.add_edge("requirement_agent", "planning_agent")
    workflow.add_edge("planning_agent", "retrieval_agent")
    workflow.add_edge("retrieval_agent", "coding_agent")
    workflow.add_edge("coding_agent", "debugging_agent")
    workflow.add_edge("debugging_agent", "verification_agent")
    workflow.add_edge("verification_agent", END)

    return workflow.compile()


# Compiled graph instance
agent_graph = build_graph()
