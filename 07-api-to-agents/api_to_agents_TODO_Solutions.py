# Class 6 — TODO Solutions
# Complete implementations of all 4 exercises from the notebook
# Each TODO builds on the base agent from the class

# ============================================================
# PREREQUISITES: Run all cells from the original class_6.ipynb
# first (Cells 0-46). These TODOs depend on:
# - client, FREE_MODEL, TOOL_MODEL (API setup)
# - TOOLS, search_wikipedia, calculate_math, get_current_date
# - CODE_TOOLS, read_file, write_file, run_python, list_files
# - REACT_SYSTEM_PROMPT, run_agent
# ============================================================

import os
import json
import re
import time
import requests
import subprocess
from openai import OpenAI


# ==============================================================
# TODO 1: CONTEXT MANAGEMENT
# ==============================================================
# Problem: Every iteration re-sends the ENTIRE conversation.
# By iteration 5, you might be sending 3000+ tokens just in
# the prompt — most of it old observations you don't need.
#
# Solution: After every N iterations, ask the LLM to summarize
# the conversation so far, then replace the full history with
# just the summary. This keeps token usage flat instead of
# growing linearly.
# ==============================================================

def summarize_history(client, messages, model):
    """
    Ask the LLM to compress the conversation history into
    a short summary. This replaces the growing message list
    with a single condensed message.
    
    Think of it like meeting notes: instead of re-reading
    the full transcript, you read a 1-paragraph summary.
    """
    # Build a summary request from the conversation
    conversation_text = ""
    for msg in messages:
        role = msg["role"] if isinstance(msg, dict) else msg.role
        content = msg.get("content", "") if isinstance(msg, dict) else (msg.content or "")
        if content:
            conversation_text += f"{role.upper()}: {content[:300]}\n"

    summary_request = [
        {"role": "system", "content": "Summarize this conversation concisely. Include: the original user goal, all facts discovered so far, tools used, and what still needs to be done. Be brief but complete."},
        {"role": "user", "content": conversation_text[:3000]}
    ]

    r = client.chat.completions.create(
        model=model,
        messages=summary_request,
        temperature=0,
        max_tokens=400,
    )

    return r.choices[0].message.content or "Summary unavailable."


def run_agent_with_context_management(
    client, user_query, model, tools_dict, system_prompt_template,
    max_iterations=10, summarize_every=3, verbose=True
):
    """
    Same ReAct agent, but with context management.
    
    After every `summarize_every` iterations, we compress
    the conversation history into a summary. This prevents
    token usage from growing linearly with each iteration.
    
    Parameters:
    - summarize_every: compress history after this many iterations
    """
    tool_desc = "\n".join(f"- {t['desc']}" for t in tools_dict.values())
    system = system_prompt_template.format(tool_descriptions=tool_desc)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_query},
    ]

    token_log = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"USER: {user_query}")
        print(f"Context management: summarize every {summarize_every} iterations")
        print(f"{'='*60}")

    for i in range(max_iterations):
        # --- CONTEXT MANAGEMENT ---
        # After every N iterations, compress the history
        if i > 0 and i % summarize_every == 0:
            if verbose:
                print(f"\n--- Compressing history (iteration {i}) ---")
                print(f"    Messages before: {len(messages)}")

            summary = summarize_history(client, messages, model)

            # Replace entire history with: system + summary + continue instruction
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"CONVERSATION SUMMARY SO FAR:\n{summary}\n\nContinue working on the original task. Use tools or give FINAL_ANSWER."},
            ]

            if verbose:
                print(f"    Messages after:  {len(messages)}")
                print(f"    Summary: {summary[:150]}...")

        if verbose:
            print(f"\n--- Iteration {i+1}/{max_iterations} ---")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=800,
        )

        tokens_in = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens
        token_log.append({"iter": i + 1, "in": tokens_in, "out": tokens_out})

        text = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": text})

        # Parse THOUGHT
        thought_match = re.search(
            r"THOUGHT:\s*(.+?)(?=ACTION:|FINAL_ANSWER:|$)", text, re.DOTALL
        )
        thought = thought_match.group(1).strip() if thought_match else ""
        if verbose and thought:
            print(f"  THOUGHT: {thought[:200]}")

        # Check for FINAL_ANSWER
        if "FINAL_ANSWER:" in text:
            answer = text.split("FINAL_ANSWER:")[-1].strip()
            if verbose:
                print(f"\n{'='*60}")
                print(f"DONE in {i+1} iteration(s)")
                print(f"{'='*60}")
                print(f"ANSWER: {answer[:500]}")
            return {"answer": answer, "iterations": i + 1, "token_log": token_log}

        # Check for ACTION
        action_match = re.search(r"ACTION:\s*(\w+)", text)
        input_match = re.search(r"ACTION_INPUT:\s*(.+?)(?:\n|$)", text, re.DOTALL)

        if not action_match:
            messages.append({
                "role": "user",
                "content": "Please respond with either ACTION + ACTION_INPUT or FINAL_ANSWER."
            })
            continue

        tool_name = action_match.group(1).strip()
        raw_input = input_match.group(1).strip() if input_match else "{}"

        # Execute tool
        if tool_name not in tools_dict:
            observation = json.dumps({"error": f"Unknown tool '{tool_name}'. Available: {list(tools_dict.keys())}"})
        else:
            try:
                if raw_input.startswith("{"):
                    args = json.loads(raw_input)
                else:
                    args = {"query": raw_input.strip("\"'")}
                observation = tools_dict[tool_name]["fn"](**args)
            except Exception as e:
                observation = json.dumps({"error": f"Failed: {e}"})

        if verbose:
            print(f"  ACTION: {tool_name}({raw_input[:80]})")
            print(f"  OBSERVATION: {observation[:200]}")

        messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})

    return {"answer": "Max iterations reached.", "iterations": max_iterations, "token_log": token_log}


# --- Test TODO 1 ---
# To run this, you need the TOOLS dict and REACT_SYSTEM_PROMPT from the original notebook.
# Uncomment below after running the original cells:
#
# result = run_agent_with_context_management(
#     client=client,
#     user_query="Who lived longer — Albert Einstein or Isaac Newton? By how many years?",
#     model=FREE_MODEL,
#     tools_dict=TOOLS,
#     system_prompt_template=REACT_SYSTEM_PROMPT,
#     max_iterations=10,
#     summarize_every=3,
#     verbose=True,
# )
#
# # Compare token usage
# print("\nTOKEN USAGE (with context management):")
# total = 0
# for t in result["token_log"]:
#     subtotal = t["in"] + t["out"]
#     total += subtotal
#     bar = "█" * (t["in"] // 50)
#     print(f"  Iter {t['iter']}: {t['in']:>5} in + {t['out']:>4} out = {subtotal:>5} | {bar}")
# print(f"\n  TOTAL: {total:,} tokens")
# print("  → Notice: prompt_tokens stays FLAT after summarization instead of growing!")


# ==============================================================
# TODO 2: PLANNING STEP
# ==============================================================
# Problem: Pure ReAct is reactive — the agent figures out the
# next step only after seeing the result of the previous one.
# For complex tasks, this can lead to wasted iterations.
#
# Solution: Add a PLAN phase before execution. The agent first
# creates a step-by-step plan, then executes each step.
# This is like writing a recipe before cooking, vs. figuring
# out each ingredient as you go.
# ==============================================================

PLAN_REACT_SYSTEM_PROMPT = """You are a helpful assistant that solves problems by first creating a plan, then executing it step by step.

You have access to these tools:
{tool_descriptions}

## Phase 1: Planning
When you receive a task, FIRST create a plan:

PLAN:
1. <first step>
2. <second step>
3. <third step>
...

## Phase 2: Execution
Then execute each step. For each step:

THOUGHT: <which plan step you're on and your reasoning>
ACTION: <tool_name>
ACTION_INPUT: <arguments as valid JSON>

Wait for the OBSERVATION before continuing to the next step.

## When Done
When all plan steps are complete:

THOUGHT: <final reasoning combining all results>
FINAL_ANSWER: <your complete answer>

## Rules
- Always start with a PLAN
- ONE action per turn
- Follow your plan, but adapt if you discover new information
- Be concise
"""


def run_plan_agent(
    client, user_query, model, tools_dict,
    max_iterations=12, verbose=True
):
    """
    Plan-then-Execute agent.
    
    Unlike pure ReAct which is reactive (think-act-observe each step),
    this agent first creates an explicit plan, then executes it.
    
    Advantage: The agent knows the full strategy upfront, so it
    doesn't waste iterations on dead ends.
    
    Disadvantage: The plan might be wrong, and the agent needs to
    adapt when reality doesn't match expectations.
    """
    tool_desc = "\n".join(f"- {t['desc']}" for t in tools_dict.values())
    system = PLAN_REACT_SYSTEM_PROMPT.format(tool_descriptions=tool_desc)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_query},
    ]

    if verbose:
        print(f"\n{'='*60}")
        print(f"USER: {user_query}")
        print(f"Mode: Plan → Execute")
        print(f"{'='*60}")

    for i in range(max_iterations):
        if verbose:
            print(f"\n--- Iteration {i+1}/{max_iterations} ---")

        response = client.chat.completions.create(
            model=model, messages=messages,
            temperature=0, max_tokens=800,
        )

        text = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": text})

        # Check for PLAN (first iteration usually)
        if "PLAN:" in text and i == 0:
            plan_text = text.split("PLAN:")[-1].strip()
            # If there's a THOUGHT after the plan, cut it off
            if "THOUGHT:" in plan_text:
                plan_text = plan_text.split("THOUGHT:")[0].strip()
            if verbose:
                print(f"  PLAN:\n{plan_text[:400]}")

        # Parse THOUGHT
        thought_match = re.search(
            r"THOUGHT:\s*(.+?)(?=ACTION:|FINAL_ANSWER:|$)", text, re.DOTALL
        )
        thought = thought_match.group(1).strip() if thought_match else ""
        if verbose and thought:
            print(f"  THOUGHT: {thought[:200]}")

        # Check for FINAL_ANSWER
        if "FINAL_ANSWER:" in text:
            answer = text.split("FINAL_ANSWER:")[-1].strip()
            if verbose:
                print(f"\n{'='*60}")
                print(f"DONE in {i+1} iteration(s)")
                print(f"{'='*60}")
                print(f"ANSWER: {answer[:500]}")
            return {"answer": answer, "iterations": i + 1, "mode": "plan-execute"}

        # Check for ACTION
        action_match = re.search(r"ACTION:\s*(\w+)", text)
        input_match = re.search(r"ACTION_INPUT:\s*(.+?)(?:\n|$)", text, re.DOTALL)

        if not action_match:
            messages.append({
                "role": "user",
                "content": "Continue with your plan. Use ACTION + ACTION_INPUT or FINAL_ANSWER."
            })
            continue

        tool_name = action_match.group(1).strip()
        raw_input = input_match.group(1).strip() if input_match else "{}"

        if tool_name not in tools_dict:
            observation = json.dumps({"error": f"Unknown tool '{tool_name}'."})
        else:
            try:
                args = json.loads(raw_input) if raw_input.startswith("{") else {"query": raw_input.strip("\"'")}
                observation = tools_dict[tool_name]["fn"](**args)
            except Exception as e:
                observation = json.dumps({"error": str(e)})

        if verbose:
            print(f"  ACTION: {tool_name}({raw_input[:80]})")
            print(f"  OBSERVATION: {observation[:200]}")

        messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})

    return {"answer": "Max iterations reached.", "iterations": max_iterations, "mode": "plan-execute"}


# --- Test TODO 2 ---
# Uncomment after running original notebook cells:
#
# # Run same task with both approaches and compare
# query = "Who was born first: the person who wrote 'Romeo and Juliet' or the person who painted the Mona Lisa? By how many years?"
#
# print("=" * 60)
# print("APPROACH 1: Pure ReAct")
# print("=" * 60)
# result_react = run_agent(query)
#
# print("\n\n")
# print("=" * 60)
# print("APPROACH 2: Plan → Execute")
# print("=" * 60)
# result_plan = run_plan_agent(client, query, FREE_MODEL, TOOLS)
#
# print(f"\n\nComparison:")
# print(f"  Pure ReAct:     {result_react['iterations']} iterations")
# print(f"  Plan→Execute:   {result_plan['iterations']} iterations")


# ==============================================================
# TODO 3: MULTI-AGENT SYSTEM
# ==============================================================
# Problem: One agent with all tools gets confused. A research
# task might accidentally trigger code tools, or a coding task
# might try to search Wikipedia for Python syntax.
#
# Solution: Create specialized agents (researcher, coder) and
# an orchestrator that routes subtasks to the right agent.
# Think of it like a company: the CEO (orchestrator) delegates
# research tasks to the researcher and coding tasks to the coder.
# ==============================================================

# --- Researcher Agent ---
# Only has Wikipedia + calculator tools
RESEARCHER_SYSTEM = """You are a research assistant. You find information and answer factual questions.

Available tools:
{tool_descriptions}

Format — to use a tool:
THOUGHT: <reasoning>
ACTION: <tool_name>
ACTION_INPUT: <args as JSON>

Format — when done:
THOUGHT: <reasoning>
FINAL_ANSWER: <your findings>

Rules: ONE action per turn. Wait for OBSERVATION. Be thorough but concise.
"""


def run_researcher(client, query, model, tools_dict, max_iterations=8, verbose=True):
    """Specialized research agent — only uses search and calculator tools."""
    tool_desc = "\n".join(f"- {t['desc']}" for t in tools_dict.items() if True)
    tool_desc = "\n".join(f"- {v['desc']}" for v in tools_dict.values())
    system = RESEARCHER_SYSTEM.format(tool_descriptions=tool_desc)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]

    if verbose:
        print(f"\n  [RESEARCHER] Task: {query[:100]}")

    for i in range(max_iterations):
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=0, max_tokens=800
        )
        text = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": text})

        if "FINAL_ANSWER:" in text:
            answer = text.split("FINAL_ANSWER:")[-1].strip()
            if verbose:
                print(f"  [RESEARCHER] Done in {i+1} iterations: {answer[:150]}")
            return answer

        action_match = re.search(r"ACTION:\s*(\w+)", text)
        input_match = re.search(r"ACTION_INPUT:\s*(.+?)(?:\n|$)", text, re.DOTALL)

        if not action_match:
            messages.append({"role": "user", "content": "Use ACTION or FINAL_ANSWER."})
            continue

        tool_name = action_match.group(1).strip()
        raw_input = input_match.group(1).strip() if input_match else "{}"

        if tool_name in tools_dict:
            try:
                args = json.loads(raw_input) if raw_input.startswith("{") else {"query": raw_input.strip("\"'")}
                observation = tools_dict[tool_name]["fn"](**args)
            except Exception as e:
                observation = json.dumps({"error": str(e)})
        else:
            observation = json.dumps({"error": f"Unknown tool: {tool_name}"})

        if verbose:
            print(f"  [RESEARCHER] {tool_name} → {observation[:100]}")

        messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})

    return "Research incomplete — max iterations reached."


# --- Coder Agent ---
# Only has file I/O and Python execution tools
CODER_SYSTEM = """You are a coding agent. You write, run, and debug Python code.

Available tools:
{tool_descriptions}

Format — to use a tool:
THOUGHT: <reasoning>
ACTION: <tool_name>
ACTION_INPUT: <args as JSON>

Format — when done:
THOUGHT: <reasoning>
FINAL_ANSWER: <what you built and its output>

Rules:
- ONE action per turn. Wait for OBSERVATION.
- Always test code after writing it.
- If a test fails, fix and retry.
"""


def run_coder(client, query, model, tools_dict, max_iterations=12, verbose=True):
    """Specialized coding agent — only uses file I/O and Python execution."""
    tool_desc = "\n".join(f"- {v['desc']}" for v in tools_dict.values())
    system = CODER_SYSTEM.format(tool_descriptions=tool_desc)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]

    if verbose:
        print(f"\n  [CODER] Task: {query[:100]}")

    for i in range(max_iterations):
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=0, max_tokens=1500
        )
        text = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": text})

        if "FINAL_ANSWER:" in text:
            answer = text.split("FINAL_ANSWER:")[-1].strip()
            if verbose:
                print(f"  [CODER] Done in {i+1} iterations: {answer[:150]}")
            return answer

        action_match = re.search(r"ACTION:\s*(\w+)", text)
        input_match = re.search(r"ACTION_INPUT:\s*(.+?)(?:\nTHOUGHT|\nACTION|\nFINAL|$)", text, re.DOTALL)

        if not action_match:
            messages.append({"role": "user", "content": "Use ACTION or FINAL_ANSWER."})
            continue

        tool_name = action_match.group(1).strip()
        raw_input = input_match.group(1).strip() if input_match else "{}"

        if tool_name in tools_dict:
            try:
                args = json.loads(raw_input)
                observation = tools_dict[tool_name]["fn"](**args)
            except json.JSONDecodeError:
                try:
                    observation = tools_dict[tool_name]["fn"](raw_input.strip("\"'"))
                except Exception as e:
                    observation = json.dumps({"error": str(e)})
            except Exception as e:
                observation = json.dumps({"error": str(e)})
        else:
            observation = json.dumps({"error": f"Unknown tool: {tool_name}"})

        if verbose:
            print(f"  [CODER] {tool_name} → {observation[:120]}")

        messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})

    return "Coding incomplete — max iterations reached."


# --- Orchestrator ---
# The "CEO" that decides which agent handles which subtask

ORCHESTRATOR_SYSTEM = """You are a task orchestrator. You break complex tasks into subtasks and delegate them to specialized agents.

Available agents:
- RESEARCHER: Can search Wikipedia and do math. Use for factual questions, lookups, comparisons.
- CODER: Can write files, run Python, list directories. Use for coding, data processing, file tasks.

For each subtask, respond in this format:

THOUGHT: <your reasoning about what needs to happen>
DELEGATE: <RESEARCHER or CODER>
SUBTASK: <clear instruction for that agent>

After all subtasks are done, combine the results:

THOUGHT: <combine all findings>
FINAL_ANSWER: <complete answer to the user>

Rules:
- Break complex tasks into 2-4 subtasks max
- ONE delegation per turn
- Wait for the agent's result before continuing
"""


def run_multi_agent(
    client, user_query, model,
    research_tools, code_tools,
    max_iterations=8, verbose=True
):
    """
    Multi-agent orchestrator.
    
    The orchestrator LLM decides whether each subtask needs
    a researcher or a coder, then delegates accordingly.
    Results from each agent are fed back to the orchestrator.
    
    This is how real multi-agent systems work:
    - CrewAI, AutoGen, LangGraph all follow this pattern
    - An orchestrator routes to specialized agents
    - Each agent has its own tools and system prompt
    """
    messages = [
        {"role": "system", "content": ORCHESTRATOR_SYSTEM},
        {"role": "user", "content": user_query},
    ]

    if verbose:
        print(f"\n{'='*60}")
        print(f"MULTI-AGENT ORCHESTRATOR")
        print(f"Task: {user_query}")
        print(f"{'='*60}")

    for i in range(max_iterations):
        if verbose:
            print(f"\n--- Orchestrator iteration {i+1} ---")

        response = client.chat.completions.create(
            model=model, messages=messages, temperature=0, max_tokens=600
        )
        text = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": text})

        # Parse thought
        thought_match = re.search(r"THOUGHT:\s*(.+?)(?=DELEGATE:|FINAL_ANSWER:|$)", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        if verbose and thought:
            print(f"  ORCHESTRATOR THOUGHT: {thought[:200]}")

        # Check for FINAL_ANSWER
        if "FINAL_ANSWER:" in text:
            answer = text.split("FINAL_ANSWER:")[-1].strip()
            if verbose:
                print(f"\n{'='*60}")
                print(f"MULTI-AGENT COMPLETE in {i+1} orchestrator iterations")
                print(f"{'='*60}")
                print(f"ANSWER: {answer[:500]}")
            return {"answer": answer, "iterations": i + 1}

        # Check for DELEGATE
        delegate_match = re.search(r"DELEGATE:\s*(RESEARCHER|CODER)", text, re.IGNORECASE)
        subtask_match = re.search(r"SUBTASK:\s*(.+?)(?:\n|$)", text, re.DOTALL)

        if not delegate_match:
            messages.append({"role": "user", "content": "Use DELEGATE + SUBTASK or FINAL_ANSWER."})
            continue

        agent_type = delegate_match.group(1).strip().upper()
        subtask = subtask_match.group(1).strip() if subtask_match else user_query

        if verbose:
            print(f"  DELEGATING to {agent_type}: {subtask[:100]}")

        # Route to the right agent
        if agent_type == "RESEARCHER":
            result = run_researcher(client, subtask, model, research_tools, verbose=verbose)
        elif agent_type == "CODER":
            result = run_coder(client, subtask, model, code_tools, verbose=verbose)
        else:
            result = f"Unknown agent type: {agent_type}"

        # Feed result back to orchestrator
        messages.append({
            "role": "user",
            "content": f"AGENT RESULT ({agent_type}):\n{result[:1000]}"
        })

    return {"answer": "Max iterations reached.", "iterations": max_iterations}


# --- Test TODO 3 ---
# Uncomment after running original notebook cells:
#
# # Research tools (subset)
# RESEARCH_TOOLS = {
#     "search_wikipedia": TOOLS["search_wikipedia"],
#     "calculate": TOOLS["calculate"],
#     "get_current_date": TOOLS["get_current_date"],
# }
#
# result = run_multi_agent(
#     client=client,
#     user_query=(
#         "Research the population of India and Japan, then write a Python script "
#         "that creates a bar chart comparing them. Save it as 'population_chart.py' and run it."
#     ),
#     model=FREE_MODEL,
#     research_tools=RESEARCH_TOOLS,
#     code_tools=CODE_TOOLS,
#     verbose=True,
# )


# ==============================================================
# TODO 4: WEB BROWSING
# ==============================================================
# Problem: The agent can only search Wikipedia. Real tasks
# often require fetching data from arbitrary web pages.
#
# Solution: Add a fetch_webpage(url) tool. This lets the agent
# follow links and read web content.
#
# New failure modes to watch for:
# 1. Pages that require JavaScript (our fetcher gets raw HTML)
# 2. Very long pages that blow up the context window
# 3. Agent getting lost following links endlessly
# 4. Paywalled or blocked content
# 5. Malicious URLs (the agent might try to fetch anything)
# ==============================================================

def fetch_webpage(url):
    """
    Fetch a webpage and return its text content.
    
    This is a simplified web browser for the agent.
    We strip HTML tags and return plain text, truncated
    to prevent context window blowup.
    
    Failure modes this introduces:
    - JS-rendered pages return empty/useless HTML
    - Very long pages get truncated, losing important info
    - Rate limiting / blocking by websites
    - Agent can get stuck in a link-following loop
    """
    try:
        # Safety: only allow http/https
        if not url.startswith(("http://", "https://")):
            return json.dumps({"error": "Only http:// and https:// URLs allowed."})

        headers = {
            "User-Agent": "Mozilla/5.0 (educational-agent-bot)"
        }
        r = requests.get(url, timeout=15, headers=headers)
        r.raise_for_status()

        content = r.text

        # Strip HTML tags (very basic — a real agent would use BeautifulSoup)
        import re as re_mod
        # Remove script and style blocks entirely
        content = re_mod.sub(r'<script[^>]*>.*?</script>', '', content, flags=re_mod.DOTALL)
        content = re_mod.sub(r'<style[^>]*>.*?</style>', '', content, flags=re_mod.DOTALL)
        # Remove all other HTML tags
        content = re_mod.sub(r'<[^>]+>', ' ', content)
        # Collapse whitespace
        content = re_mod.sub(r'\s+', ' ', content).strip()

        # Truncate to prevent context blowup (critical!)
        max_chars = 2000
        truncated = len(content) > max_chars

        return json.dumps({
            "url": url,
            "content": content[:max_chars],
            "truncated": truncated,
            "total_chars": len(content),
        })
    except requests.exceptions.Timeout:
        return json.dumps({"error": f"Timeout fetching {url}"})
    except requests.exceptions.HTTPError as e:
        return json.dumps({"error": f"HTTP error: {e}"})
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch: {str(e)}"})


# Add it to our tool registry
WEB_TOOLS = {
    "search_wikipedia": {
        "fn": lambda query: (
            requests.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}",
                timeout=10
            ).json()
            if requests.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}",
                timeout=10
            ).status_code == 200
            else {"error": "Not found"}
        ),  # Simplified — use the original search_wikipedia from the notebook
        "desc": "search_wikipedia(query: str) — Search Wikipedia for a topic."
    },
    "fetch_webpage": {
        "fn": fetch_webpage,
        "desc": "fetch_webpage(url: str) — Fetch a webpage URL and return its text content. Use full URLs like 'https://example.com'."
    },
    "calculate": {
        "fn": lambda expression: json.dumps({"result": round(eval(expression), 6)}) if all(c in "0123456789+-*/.() eE" for c in expression) else json.dumps({"error": "Invalid chars"}),
        "desc": "calculate(expression: str) — Evaluate a math expression."
    },
}

# NOTE: When using in the actual notebook, replace the lambda
# functions above with the real search_wikipedia and calculate_math
# functions from the original notebook. The lambdas are just
# placeholders so this file is self-contained.

WEB_AGENT_SYSTEM = """You are a web research assistant. You can search Wikipedia and browse web pages.

Available tools:
{tool_descriptions}

Format — to use a tool:
THOUGHT: <your reasoning>
ACTION: <tool_name>
ACTION_INPUT: <args as JSON>

Format — when done:
THOUGHT: <final reasoning>
FINAL_ANSWER: <your answer>

## Important rules
- ONE action per turn
- When using fetch_webpage, the content is truncated to 2000 chars. Focus on finding the key info quickly.
- Do NOT follow more than 3 links — summarize what you have.
- Prefer search_wikipedia for factual lookups. Use fetch_webpage only for specific URLs.
- If a page is blocked or empty, try a different approach rather than retrying the same URL.
"""


def run_web_agent(
    client, user_query, model, tools_dict,
    max_iterations=10, verbose=True
):
    """
    Agent with web browsing capability.
    
    Same ReAct pattern, but now with fetch_webpage tool.
    Watch for these new failure modes:
    
    1. CONTEXT BLOWUP: Web pages are huge. Even truncated to 2000 chars,
       3 page fetches = 6000+ chars of observations filling the context.
    
    2. RABBIT HOLES: The agent might keep following links instead of
       answering. The system prompt limits it to 3 links max.
    
    3. EMPTY PAGES: JS-rendered sites return useless HTML to our simple
       fetcher. The agent needs to recognize this and try another source.
    
    4. BLOCKED CONTENT: Many sites block automated requests. The agent
       sees an error and needs to adapt.
    """
    tool_desc = "\n".join(f"- {v['desc']}" for v in tools_dict.values())
    system = WEB_AGENT_SYSTEM.format(tool_descriptions=tool_desc)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_query},
    ]

    links_followed = 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"WEB AGENT")
        print(f"Task: {user_query}")
        print(f"{'='*60}")

    for i in range(max_iterations):
        if verbose:
            print(f"\n--- Iteration {i+1}/{max_iterations} ---")

        response = client.chat.completions.create(
            model=model, messages=messages, temperature=0, max_tokens=800
        )
        text = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": text})

        thought_match = re.search(r"THOUGHT:\s*(.+?)(?=ACTION:|FINAL_ANSWER:|$)", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        if verbose and thought:
            print(f"  THOUGHT: {thought[:200]}")

        if "FINAL_ANSWER:" in text:
            answer = text.split("FINAL_ANSWER:")[-1].strip()
            if verbose:
                print(f"\n{'='*60}")
                print(f"DONE in {i+1} iterations (followed {links_followed} links)")
                print(f"{'='*60}")
                print(f"ANSWER: {answer[:500]}")
            return {"answer": answer, "iterations": i + 1, "links_followed": links_followed}

        action_match = re.search(r"ACTION:\s*(\w+)", text)
        input_match = re.search(r"ACTION_INPUT:\s*(.+?)(?:\n|$)", text, re.DOTALL)

        if not action_match:
            messages.append({"role": "user", "content": "Use ACTION or FINAL_ANSWER."})
            continue

        tool_name = action_match.group(1).strip()
        raw_input = input_match.group(1).strip() if input_match else "{}"

        # Track link following
        if tool_name == "fetch_webpage":
            links_followed += 1
            if links_followed > 3:
                observation = json.dumps({
                    "error": "Link limit reached (3 max). Please summarize what you have and give FINAL_ANSWER."
                })
                if verbose:
                    print(f"  BLOCKED: Link limit reached ({links_followed})")
                messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})
                continue

        if tool_name in tools_dict:
            try:
                args = json.loads(raw_input) if raw_input.startswith("{") else {"query": raw_input.strip("\"'")}
                # Handle fetch_webpage which takes 'url' not 'query'
                if tool_name == "fetch_webpage" and "query" in args and "url" not in args:
                    args = {"url": args["query"]}
                observation = tools_dict[tool_name]["fn"](**args)
            except Exception as e:
                observation = json.dumps({"error": str(e)})
        else:
            observation = json.dumps({"error": f"Unknown tool: {tool_name}"})

        if verbose:
            print(f"  ACTION: {tool_name}({raw_input[:80]})")
            obs_preview = observation[:200] + "..." if len(observation) > 200 else observation
            print(f"  OBSERVATION: {obs_preview}")

        messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})

    return {"answer": "Max iterations reached.", "iterations": max_iterations, "links_followed": links_followed}


# --- Test TODO 4 ---
# Uncomment after running original notebook cells:
#
# # First, fix WEB_TOOLS to use real functions from the notebook:
# WEB_TOOLS = {
#     "search_wikipedia": TOOLS["search_wikipedia"],
#     "calculate": TOOLS["calculate"],
#     "fetch_webpage": {"fn": fetch_webpage, "desc": "fetch_webpage(url: str) — Fetch a webpage and return its text content."},
# }
#
# result = run_web_agent(
#     client=client,
#     user_query="What is the latest Python version? Check https://www.python.org and tell me.",
#     model=FREE_MODEL,
#     tools_dict=WEB_TOOLS,
#     verbose=True,
# )


# ==============================================================
# SUMMARY OF ALL 4 TODOS
# ==============================================================
#
# TODO 1 — Context Management:
#   Added summarize_history() that compresses conversation every
#   N iterations. Token usage stays flat instead of growing.
#   Key insight: summarization is lossy but keeps costs manageable.
#
# TODO 2 — Planning Step:
#   Added PLAN phase before execution. Agent writes a plan first,
#   then follows it step by step. Often uses fewer iterations
#   than pure ReAct because it doesn't waste time exploring.
#   Key insight: planning is "thinking before doing."
#
# TODO 3 — Multi-Agent:
#   Created Researcher (search + math) and Coder (files + Python)
#   with an Orchestrator that routes subtasks. Each agent has its
#   own system prompt and tool set.
#   Key insight: specialization > generalization for complex tasks.
#
# TODO 4 — Web Browsing:
#   Added fetch_webpage(url) tool with safety limits (3 link max,
#   2000 char truncation). New failure modes: JS-rendered pages,
#   context blowup, rabbit holes, blocked content.
#   Key insight: more tools = more failure modes to handle.
# ==============================================================
