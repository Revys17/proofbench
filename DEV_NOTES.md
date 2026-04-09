This file contains the description of the development process for this project.

# LLM Usage

I worked with Claude Opus 4.6 and Gemini 3.1 Pro to develop the project plan, and used Claude Code heavily during implementation.

# Design

I decided to go with the simplest design that would get results as quickly as possible:
* Custom prompting for each model (generator and solver)
* Custom LLM client and agentic loop rather than something like `claude -p`
* Local Lean install rather than Docker

# Results

TODO - once we have some nice JSONs, fill in the results here.

For full results, see the run results JSON files under `results/`.

# Limitations

* We limit ourselves to only the Claude family of models - OpenAI support is implemented but not tested because I don't have a working OpenAI API key atm.
* The generator model budget was somewhat underspecified in the problem statement. I interpreted this as **the generator gets some number of theorem submissions**, after which the loop ends. If instead there should be a token limit, we would have to implement this.
* The eval results depend heavily on which models you choose to use as solvers, and the results don't mean much when any of the solvers are generally stronger than the generator.


# Extensions

* Token-based generator/solver budgets rather than attempt-based budgets
* Larger runs take a while - a run dashboard would help with monitoring run status
* Creating some sort of ELO score between models that allows stack ranking seems like a more useful metric than the gap score, which depends more on which models were included as solvers.
* We could intentionally diversify the math domains that the generators operate in via specialized prompting to check whether math capabilities are jagged across domains.
