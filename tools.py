"""Custom tools for the multi-agent PLC system."""

import json
from typing import List, Optional
from duckduckgo_search import DDGS
from pypdf import PdfReader
import io


def web_search(query: str, max_results: int = 5) -> List[dict]:
    """Search the web for PLC/automation documentation."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return [{"title": r.get("title", ""), "body": r.get("body", ""), "href": r.get("href", "")} for r in results]
    except Exception as e:
        return [{"error": str(e)}]


def parse_pdf(pdf_bytes: bytes) -> str:
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages[:50]:  # limit to 50 pages
            text += page.extract_text() or ""
            text += "\n\n"
        return text.strip()
    except Exception as e:
        return f"Error parsing PDF: {e}"


def validate_structured_text(code: str) -> dict:
    """Basic IEC 61131-3 Structured Text validation."""
    issues = []
    lines = code.split("\n")

    open_blocks = {
        "IF": 0, "CASE": 0, "FOR": 0,
        "WHILE": 0, "REPEAT": 0,
        "FUNCTION_BLOCK": 0, "FUNCTION": 0, "PROGRAM": 0,
    }

    for i, line in enumerate(lines, 1):
        stripped = line.strip().upper()

        # Track block openings
        for keyword in open_blocks:
            if stripped.startswith(keyword) and not stripped.startswith(f"END_{keyword}"):
                open_blocks[keyword] += 1
            elif stripped.startswith(f"END_{keyword}"):
                open_blocks[keyword] -= 1

        # Check for missing semicolons on assignment lines
        if ":=" in line and not stripped.endswith(";") and not stripped.endswith("THEN") and stripped:
            issues.append({
                "line": i,
                "severity": "error",
                "message": f"Missing semicolon at end of assignment",
            })

    # Check unclosed blocks
    for keyword, count in open_blocks.items():
        if count > 0:
            issues.append({
                "severity": "error",
                "message": f"Unclosed {keyword} block ({count} unclosed)",
            })

    return {"valid": len(issues) == 0, "issues": issues}


def extract_io_tags(code: str) -> List[dict]:
    """Extract I/O variable declarations with AT bindings from ST code."""
    tags = []
    for line in code.split("\n"):
        stripped = line.strip()
        if "AT %" in stripped.upper():
            parts = stripped.split(":")
            if len(parts) >= 2:
                var_name = parts[0].strip()
                rest = ":".join(parts[1:])
                tags.append({"name": var_name, "declaration": rest.strip().rstrip(";")})
    return tags


def check_safety_patterns(code: str) -> List[dict]:
    """Check for common safety patterns in PLC code."""
    checks = []
    code_upper = code.upper()

    # Emergency stop check
    has_estop = any(kw in code_upper for kw in ["ESTOP", "E_STOP", "EMERGENCY", "EMERGENCYSTOP"])
    checks.append({
        "check": "Emergency stop logic present",
        "passed": has_estop,
        "notes": "E-stop handling found" if has_estop else "No emergency stop logic detected"
    })

    # Watchdog/heartbeat
    has_watchdog = any(kw in code_upper for kw in ["WATCHDOG", "HEARTBEAT", "TIMEOUT"])
    checks.append({
        "check": "Watchdog/timeout mechanism",
        "passed": has_watchdog,
        "notes": "Timeout/watchdog found" if has_watchdog else "Consider adding watchdog timers"
    })

    # State machine pattern
    has_states = "CASE" in code_upper and ("STATE" in code_upper or "E_STATE" in code_upper)
    checks.append({
        "check": "State machine pattern used",
        "passed": has_states,
        "notes": "CASE-based state machine found" if has_states else "No state machine pattern detected"
    })

    return checks
