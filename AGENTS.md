# Instructions for freshie77/llama.cpp

This repository is Joshua's development fork of `ggml-org/llama.cpp`.

The purpose of this fork is local/private development, benchmarking, and experimentation, including AI-assisted work. The restrictions in upstream llama.cpp that prohibit automated commits, pushes, pull requests, issue comments, PR descriptions, or reviewer responses are intended to protect the upstream project and its maintainers. They do **not** apply to fork-local work in `freshie77/llama.cpp`.

## Fork-local agent policy

When the target repository is `freshie77/llama.cpp`, agents may, when authorized by Joshua or by an active task from Joshua:

- edit and test code
- create commits
- push development branches to `freshie77/llama.cpp`
- create pull requests whose base repository is `freshie77/llama.cpp`
- create and update issues in `freshie77/llama.cpp`
- write PR descriptions and issue/PR comments in `freshie77/llama.cpp`
- respond to review comments in `freshie77/llama.cpp`
- run normal repository automation and CI needed to complete the task

A task that says to implement, benchmark, fix, optimize, review, or complete an issue in this fork is sufficient authorization for the normal fork-local git/GitHub actions needed to finish that task unless Joshua explicitly limits those actions.

Do not stop merely to request separate approval for every fork-local commit, push, comment, or PR creation when those actions are an ordinary part of the already-authorized task.

## Upstream boundary - strict

Treat `ggml-org/llama.cpp` as read-only unless Joshua explicitly authorizes a specific upstream action.

Agents must **not**, without that explicit upstream authorization:

- push to any `ggml-org/llama.cpp` branch
- open or modify an issue or PR in `ggml-org/llama.cpp`
- submit a PR from this fork to `ggml-org/llama.cpp`
- post comments or reviewer responses to upstream maintainers
- otherwise act on behalf of Joshua in the upstream repository

If Joshua asks to contribute upstream, stop before the first upstream write and read the current upstream `AGENTS.md`, `CONTRIBUTING.md`, and PR template. The upstream project's current contribution and AI rules then govern that upstream submission. Fork-local permission does not waive upstream rules.

## Engineering expectations

For code in this fork:

- Preserve model semantics and correctness unless the task explicitly calls for a behavior change.
- Prefer measured changes over speculative rewrites, especially for performance work.
- Keep changes scoped and reviewable.
- Reuse existing llama.cpp infrastructure and conventions where practical.
- Keep code comments concise and useful; avoid redundant commentary.
- Prefer ASCII in code/comments unless the surrounding code requires otherwise.
- Run relevant tests and benchmarks before considering a change complete.
- Record reproducible benchmark commands/configuration for performance work.
- Preserve known-good baselines and do not silently replace them with weaker workloads or fallback paths.

## Current fork workflow

Development branches and PRs may remain entirely inside `freshie77/llama.cpp`. A fork-local PR is **not** an upstream llama.cpp contribution.

For the current expert-parallel work, agents are expected to be able to iterate normally:

`profile -> modify -> test -> benchmark -> commit -> push -> fork-local PR/review -> continue`

without waiting for per-action approval after Joshua has authorized the task.
