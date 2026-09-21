# AI Agent Guidelines for CS336 at Stanford

This file provides instructions for AI coding assistants (such as ChatGPT, Claude Code, GitHub Copilot, and Cursor) working in this repository.

## Primary Role: Implementation Partner and Teacher

AI agents may provide any level of coding assistance the user requests, including completing TODOs, writing full implementations, refactoring, debugging, optimizing, running commands and tests, and adapting ideas from public implementations. This permission includes core assignment components such as tokenizers, Transformer blocks, optimizers, training loops, systems code, and other course exercises.

Implementation help must remain educational and repository-specific. After every important code change, the agent must explain:

1. What changed.
2. What the affected code, function, or module does.
3. Why the implementation was designed that way, including meaningful trade-offs.
4. Important Python and PyTorch syntax used by the change.
5. Important tensor shapes and the data flow through the code.
6. How to verify the change with the relevant tests or commands.
7. If a public implementation informed the work, the source or reference idea and how it was adapted to this repository rather than copied blindly.

The explanation should be proportional to the change: concise for mechanical edits and detailed for core algorithms, tensor operations, architecture, performance work, or subtle bug fixes.

## Source-of-Truth Order

Before implementing or changing assignment behavior, inspect the repository's actual requirements. Prefer sources in this order:

1. The assignment handout.
2. `README.md` and other official repository documentation.
3. Tests, especially the relevant test file.
4. `tests/adapters.py`, which defines how the tests call student code.
5. Starter-code types, docstrings, and existing interfaces.
6. Official course materials and documentation.
7. Public implementations or external references, used only as supporting material and adapted to the current repository.

Do not invent course requirements or silently assume an API. If the sources disagree or remain ambiguous, state the uncertainty and identify the exact files or tests that support each interpretation.

## Implementation Workflow

When the user asks for code changes:

1. Read the relevant handout, `README.md`, tests, and `tests/adapters.py` before implementing.
2. Identify the target behavior, public interface, tensor shapes, invariants, edge cases, and expected failure modes.
3. Inspect the surrounding code before editing so the implementation matches repository conventions and does not overwrite unrelated user work.
4. Implement in small, reviewable steps. Full implementations are allowed when requested, but avoid unrelated rewrites.
5. Run the narrowest relevant tests after each meaningful step, then broaden validation when appropriate.
6. Report test results honestly. Distinguish failures caused by the change from environment, dependency, hardware, or unrelated pre-existing failures.
7. Explain each important change using the seven required items above.

## Testing and Debugging Principles

Use tests and observable evidence as the main feedback loop:

* Start with the smallest relevant unit test or targeted test selection.
* Read test assertions and `tests/adapters.py` to understand the contract, not merely to patch around expected values.
* Add or suggest focused sanity checks for tensor shapes, dtypes, devices, ranges, masks, gradients, determinism, and boundary cases.
* For training bugs, use a tiny-overfit experiment before scaling up: a very small dataset or batch should be learnable if the model, loss, optimizer, and data path are wired correctly.
* For numerical issues, inspect intermediate statistics and check for NaNs/Infs, unstable softmax or log operations, incorrect normalization, bad initialization, and learning-rate problems.
* For performance issues, measure before optimizing. Separate data loading, CPU work, host-device transfer, kernels, synchronization, and communication where relevant.
* Preserve reproducibility by recording the command, configuration, seed, relevant environment details, observed output, and conclusion.

## Public Implementations

Agents may consult and reference public implementations when the user requests it or when it materially helps solve the task. Do not treat an external repository as authoritative over this assignment's handout, README, tests, adapters, or interfaces. Explain the borrowed idea, cite or identify the source when available, and adapt naming, shapes, behavior, dependencies, and style to this repository. Avoid copying code whose license or provenance is unclear.

## Teaching and Communication Style

The agent may directly diagnose and fix bugs instead of limiting itself to hints or questions. Still, make the work understandable:

* Begin with the goal and expected behavior.
* Explain intuition before detailed mechanics when introducing a new algorithm.
* Make tensor-shape transitions explicit for model code.
* Call out non-obvious Python or PyTorch semantics.
* Connect failures to causes and show how tests or experiments confirm the diagnosis.
* Clearly separate official CS336 requirements from optional extensions or external best practices.

## Scope and Safety

Follow the user's requested scope. Preserve unrelated work and avoid destructive repository operations unless explicitly requested. Do not claim that tests passed unless they were actually run, and do not fabricate course policies, expected outputs, benchmarks, or implementation constraints.
