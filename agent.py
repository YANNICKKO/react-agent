#!/usr/bin/env python3
"""
ReAct Agent — Reason + Act loop with tool use
Uses the Anthropic API with tool_use to autonomously solve tasks.

Architecture:
  1. User gives a task
  2. Agent reasons about what to do (Thought)
  3. Agent picks a tool (Action)
  4. Tool runs and returns result (Observation)
  5. Loop until the agent calls `finish`

Tools available:
  - web_search      : search the web for real-time info
  - calculator      : evaluate math expressions safely
  - get_weather     : fetch current weather for any city (uses open-meteo, no key needed)
  - get_current_time: return the current date/time for any timezone
  - save_report     : write a markdown report to disk
  - finish          : return the final answer to the user
"""

import anthropic
import json
import math
import os
import re
import sys
import urllib.request
import urllib.parse
from datetime import datetime

# ── Colour helpers ────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
MAGENTA= "\033[95m"
RED    = "\033[91m"
DIM    = "\033[2m"

def c(text, colour): return f"{colour}{text}{RESET}"

# ── Tool definitions (sent to the API) ───────────────────────────────────────
TOOLS = [
    {
        "name": "web_search",
        "description": (
            "Search the web for up-to-date information. "
            "Use this when you need facts, news, prices, or anything that may have changed recently. "
            "Returns a list of relevant snippets."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query to look up."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculator",
        "description": (
            "Evaluate a mathematical expression and return the result. "
            "Supports standard Python math: +, -, *, /, **, %, sqrt(), log(), sin(), cos(), etc. "
            "Always use this instead of doing mental arithmetic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "A valid Python math expression, e.g. '2 ** 10' or 'sqrt(144)'."}
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_weather",
        "description": (
            "Get the current weather for any city in the world. "
            "Returns temperature, wind speed, and weather condition. "
            "Use this whenever the user asks about weather or needs it to answer a question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'London' or 'New York'."},
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature units. Default: celsius."
                },
            },
            "required": ["city"],
        },
    },
    {
        "name": "get_current_time",
        "description": (
            "Return the current date and time. "
            "Optionally specify a timezone (e.g. 'Europe/London', 'America/New_York'). "
            "Use this whenever the task involves the current date, time, or day of the week."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone name, e.g. 'UTC', 'America/New_York'. Defaults to local system time."
                },
            },
            "required": [],
        },
    },
    {
        "name": "save_report",
        "description": "Save a markdown-formatted report to a local file. Use this to persist the final output.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Filename without extension, e.g. 'research_report'."},
                "content":  {"type": "string", "description": "Full markdown content to write to the file."},
            },
            "required": ["filename", "content"],
        },
    },
    {
        "name": "finish",
        "description": "Call this when you have fully completed the task. Provide the final answer to show the user.",
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "The complete final answer or summary for the user."}
            },
            "required": ["answer"],
        },
    },
]

# ── Tool implementations ──────────────────────────────────────────────────────
def run_web_search(query: str) -> str:
    """Calls the Brave Search API (free tier) or falls back to a mock."""
    api_key = os.environ.get("BRAVE_API_KEY")
    if api_key:
        try:
            url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode({"q": query, "count": 5})
            req = urllib.request.Request(url, headers={"Accept": "application/json", "X-Subscription-Token": api_key})
            with urllib.request.urlopen(req, timeout=8) as r:
                data = json.loads(r.read())
            results = data.get("web", {}).get("results", [])
            snippets = [f"• {r['title']}: {r['description']}" for r in results[:5]]
            return "\n".join(snippets) if snippets else "No results found."
        except Exception as e:
            return f"[Search error: {e}]"
    else:
        return (
            f"[Simulated search results for: '{query}']\n"
            f"• Result 1: Overview of {query} — Wikipedia gives a comprehensive introduction.\n"
            f"• Result 2: Latest news on {query} from Reuters (2026).\n"
            f"• Result 3: {query} explained simply — a beginner-friendly guide.\n"
            "Tip: Set BRAVE_API_KEY in your environment for real search results."
        )


def run_calculator(expression: str) -> str:
    """Safely evaluate a math expression."""
    safe_globals = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    safe_globals.update({"__builtins__": {}})
    try:
        result = eval(expression, safe_globals)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


def run_get_weather(city: str, units: str = "celsius") -> str:
    """Fetch real current weather via open-meteo (no API key required)."""
    try:
        # Step 1: geocode city → lat/lon using open-meteo's geocoding API
        geo_url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode({
            "name": city, "count": 1, "language": "en", "format": "json"
        })
        with urllib.request.urlopen(geo_url, timeout=8) as r:
            geo = json.loads(r.read())

        if not geo.get("results"):
            return f"Could not find location: '{city}'"

        result = geo["results"][0]
        lat, lon = result["latitude"], result["longitude"]
        place = result.get("name", city)
        country = result.get("country", "")

        # Step 2: fetch weather
        temp_unit = "fahrenheit" if units == "fahrenheit" else "celsius"
        wx_url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,wind_speed_10m,weather_code",
            "temperature_unit": temp_unit,
            "wind_speed_unit": "kmh",
        })
        with urllib.request.urlopen(wx_url, timeout=8) as r:
            wx = json.loads(r.read())

        cur = wx["current"]
        temp = cur["temperature_2m"]
        wind = cur["wind_speed_10m"]
        code = cur["weather_code"]

        # WMO weather code → description
        WMO = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Icy fog", 51: "Light drizzle", 53: "Drizzle",
            55: "Heavy drizzle", 61: "Light rain", 63: "Rain", 65: "Heavy rain",
            71: "Light snow", 73: "Snow", 75: "Heavy snow", 80: "Rain showers",
            85: "Snow showers", 95: "Thunderstorm", 99: "Thunderstorm with hail",
        }
        condition = WMO.get(code, f"Weather code {code}")
        unit_sym = "°F" if units == "fahrenheit" else "°C"

        return (
            f"Weather in {place}, {country}:\n"
            f"  🌡  Temperature : {temp}{unit_sym}\n"
            f"  💨  Wind speed  : {wind} km/h\n"
            f"  🌤  Condition   : {condition}"
        )
    except Exception as e:
        return f"[Weather error: {e}]"


def run_get_current_time(timezone: str = "") -> str:
    """Return the current date and time, optionally in a given IANA timezone."""
    try:
        if timezone:
            # Use zoneinfo (stdlib from Python 3.9+) if available
            try:
                from zoneinfo import ZoneInfo
                now = datetime.now(ZoneInfo(timezone))
                return now.strftime(f"%A, %d %B %Y %H:%M:%S ({timezone})")
            except Exception:
                pass
        now = datetime.now()
        label = timezone if timezone else "local time"
        return now.strftime(f"%A, %d %B %Y %H:%M:%S") + f" ({label})"
    except Exception as e:
        return f"[Time error: {e}]"


def run_save_report(filename: str, content: str) -> str:
    """Write markdown content to a file."""
    fname = re.sub(r"[^\w\-]", "_", filename) + ".md"
    path = os.path.join(os.getcwd(), fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Report saved to: {path}"


# ── Tool dispatcher ───────────────────────────────────────────────────────────
def dispatch_tool(name: str, inputs: dict) -> str:
    if name == "web_search":
        return run_web_search(inputs["query"])
    elif name == "calculator":
        return run_calculator(inputs["expression"])
    elif name == "get_weather":
        return run_get_weather(inputs["city"], inputs.get("units", "celsius"))
    elif name == "get_current_time":
        return run_get_current_time(inputs.get("timezone", ""))
    elif name == "save_report":
        return run_save_report(inputs["filename"], inputs["content"])
    elif name == "finish":
        return inputs["answer"]  # handled specially in the loop
    return f"Unknown tool: {name}"


# ── Pretty printing ───────────────────────────────────────────────────────────
def print_header():
    print(c("╔══════════════════════════════════════════════════╗", CYAN))
    print(c("║          🤖  ReAct Agent  (Reason + Act)         ║", CYAN))
    print(c("║  web_search · calculator · weather · time · more ║", CYAN))
    print(c("╚══════════════════════════════════════════════════╝", CYAN))
    print()

def print_step(step: int, kind: str, content: str, colour: str):
    label = f"[Step {step}] {kind}"
    print(c(f"\n{'─'*52}", DIM))
    print(c(f"  {label}", colour + BOLD))
    print(c("─"*52, DIM))
    for line in content.splitlines():
        print(f"  {line}")

def print_tool_call(name: str, inputs: dict):
    print(c(f"\n  🔧 Tool call → {name}", YELLOW + BOLD))
    for k, v in inputs.items():
        preview = str(v)[:120] + ("…" if len(str(v)) > 120 else "")
        print(f"     {c(k, DIM)}: {preview}")

def print_observation(result: str):
    print(c("\n  📋 Observation:", GREEN + BOLD))
    for line in result.splitlines()[:10]:
        print(f"     {line}")
    if result.count("\n") > 10:
        print(c("     … (truncated)", DIM))


# ── Main agent loop ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a ReAct agent. You solve tasks by alternating between reasoning and acting.

For every response you MUST use one of the provided tools — never reply with plain text only.

Strategy:
1. Think step-by-step about what you need to do.
2. Use web_search for facts or recent information.
3. Use calculator for any maths.
4. Use get_weather when the task involves weather for a city.
5. Use get_current_time when the task involves the current date or time.
6. Use save_report if the task asks you to produce a written document.
7. Call finish when the task is fully complete.

Be thorough. Use multiple tool calls across turns if needed. Do not guess facts — search for them."""


def run_agent(task: str, max_steps: int = 15):
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": task}]

    print(c(f"\n  📝 Task: {task}", MAGENTA + BOLD))
    print(c(f"  Max steps: {max_steps}\n", DIM))

    step = 0
    while step < max_steps:
        step += 1

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        text_parts = [b.text for b in response.content if b.type == "text" and b.text.strip()]
        tool_uses  = [b for b in response.content if b.type == "tool_use"]

        if text_parts:
            print_step(step, "💭 Thought", "\n".join(text_parts), CYAN)

        if not tool_uses:
            print(c("\n  ⚠  Agent returned no tool calls. Ending loop.", RED))
            break

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tu in tool_uses:
            print_tool_call(tu.name, tu.input)

            if tu.name == "finish":
                final_answer = tu.input["answer"]
                print(c("\n╔══════════════════════════════════════════════════╗", GREEN))
                print(c("║                ✅  Task Complete                  ║", GREEN))
                print(c("╚══════════════════════════════════════════════════╝", GREEN))
                print(f"\n{final_answer}\n")
                return final_answer

            result = dispatch_tool(tu.name, tu.input)
            print_observation(result)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})

        if response.stop_reason == "end_turn" and not tool_uses:
            break

    print(c("\n  ⚠  Reached max steps without finish.", RED))


# ── Entry point ───────────────────────────────────────────────────────────────
EXAMPLE_TASKS = [
    "What is 2 raised to the power of 32, and how many gigabytes is that if treated as bytes?",
    "What's the current weather in Tokyo and what time is it there right now?",
    "Research the latest trends in AI agents and save a short markdown report called 'ai_agents_report'.",
    "Calculate the compound interest on $10,000 invested for 5 years at 7% annual interest.",
]

def run_demo():
    """Simulate a full agent run without calling the API — for testing/demo purposes."""
    print(c("  ── DEMO MODE (no API key needed) ──\n", YELLOW + BOLD))
    task = "What's the weather in Paris and what time is it there?"
    print(c(f"  📝 Task: {task}", MAGENTA + BOLD))

    import time

    steps = [
        ("thought", "I need to get the weather in Paris and the current time there. Let me start with the weather.", CYAN),
        ("tool",    "get_weather", {"city": "Paris", "units": "celsius"}),
        ("obs",     run_get_weather("Paris", "celsius")),
        ("thought", "Got the weather. Now let me get the current time in Paris.", CYAN),
        ("tool",    "get_current_time", {"timezone": "Europe/Paris"}),
        ("obs",     run_get_current_time("Europe/Paris")),
        ("finish",  None),
    ]

    for i, step in enumerate(steps, 1):
        time.sleep(0.3)
        kind = step[0]
        if kind == "thought":
            print_step(i, "💭 Thought", step[1], step[2])
        elif kind == "tool":
            print_tool_call(step[1], step[2])
        elif kind == "obs":
            print_observation(step[1])
        elif kind == "finish":
            print(c("\n╔══════════════════════════════════════════════════╗", GREEN))
            print(c("║                ✅  Task Complete                  ║", GREEN))
            print(c("╚══════════════════════════════════════════════════╝", GREEN))
            print("\nParis weather and local time retrieved successfully.\n")

    print()


def main():
    print_header()

    if "--demo" in sys.argv:
        run_demo()
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(c("  ⚠  ANTHROPIC_API_KEY not set.", RED + BOLD))
        print(c("  Set it with:  export ANTHROPIC_API_KEY=sk-ant-...\n", DIM))
        print(c("  Running in --demo mode instead.\n", YELLOW))
        run_demo()
        return

    if len(sys.argv) > 1 and sys.argv[1] != "--demo":
        task = " ".join(sys.argv[1:])
    else:
        print(c("  Example tasks you can try:\n", DIM))
        for i, t in enumerate(EXAMPLE_TASKS, 1):
            print(f"  {c(str(i), YELLOW)}. {t}")
        print()
        task = input(c("  Enter your task (or press Enter for example 2): ", BOLD)).strip()
        if not task:
            task = EXAMPLE_TASKS[1]

    run_agent(task)


if __name__ == "__main__":
    main()