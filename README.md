# ReAct Agent

An LLM-powered agent built with the Anthropic API using the ReAct (Reason + Act) pattern.

## Tools
- web_search — search the web
- calculator — evaluate math expressions
- get_weather — live weather for any city
- get_current_time — current time in any timezone
- save_report — write results to a markdown file

## Setup

pip install anthropic

Set your API key:
- Windows: set ANTHROPIC_API_KEY=sk-ant-...
- Mac/Linux: export ANTHROPIC_API_KEY=sk-ant-...

## Run

python agent.py

## Demo (no API key needed)

python agent.py --demo
