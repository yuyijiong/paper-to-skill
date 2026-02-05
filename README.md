# Paper to Skill

Convert AI agent research papers into **Claude Code skills**—reusable instructions and (optionally) scripts that Claude can follow. This skill is specialized for **memory and learning systems** (experience memory, RAG, continuous learning, feedback loops).

## What It Does

- **Paper + code**: Extracts the method from the paper and adapts the repository into a skill (scripts + workflow).
- **Paper only**: Extracts the algorithm and produces structured guidance and a skill outline.
- **Code only**: Reverse-engineers the workflow from the code and documents it as a skill.

The result is a skill directory with `SKILL.md`, optional `scripts/`, and `references/` (e.g. algorithm details) that you can drop into `.claude/skills/` and use in Claude Code or Cursor.

## How to Use It

### 1. Trigger the skill
First, copy this repo into your `.claude/skills/` directory.

Then, in Claude Code, trigger it by typing `/paper-to-skill` and hitting enter.

### 2. Provide input

Give Claude one or more of:

| Input | What to provide |
|-------|------------------|
| **Paper** | Attach the PDF or paste the path (e.g. `paper.pdf`). |
| **Code** | Repository path, or paste relevant files / structure. |
| **Both** | Paper + code for full conversion with scripts. |

Example prompts:

- *"Convert the attached PDF to a skill."*
- *"Create a skill from `path/to/paper.pdf` and the repo in `path/to/code/`."*
- *"Turn the algorithm in this paper into a skill; I only have the PDF."*

## Scope

- **In scope**: Papers about methods that **AI agent systems** can use (memory, retrieval, learning, feedback). The skill focuses on turning those into actionable skills.
- **Out of scope**: If the paper is not about an agent-usable method, Claude will explain instead of converting.

## Examples

The `examples/` directory contains skills generated from papers:

- **live-evo-memory** — From *Live-Evo: Online Evolution of Agentic Memory from Continuous Feedback* (paper + code).
- **lightmem** — From *LightMem: Lightweight and Efficient Memory-Augmented Generation* (paper-focused).
- **general-agentic-memory** — From *General Agentic Memory Via Deep Research* (paper-focused).

Each example includes the source PDF and the resulting `SKILL.md`.
