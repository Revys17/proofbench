This file contains the description of the development process for this project.

# LLM Usage

I worked with Claude Opus 4.6 and Gemini 3.1 Pro to develop the project plan, and used Claude Code heavily during implementation.

# Design

I decided to go with the simplest design that would get results as quickly as possible:
* Custom prompting for each model
* Custom LLM client and agentic loop rather than something like `claude -p`
* Local Lean install rather than Docker

The first hour or so was spent reading the problem statement, planning out the design, and installing dependencies.

# Results



For full results, see the run results JSON files under `results/`.

# Limitations

* We limit ourselves to only the Claude family of models.


# Extensions

*
