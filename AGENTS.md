# Instructions for the VITRIOL llama.cpp fork

> This is a **private fork** of `ggml-org/llama.cpp` (`vitriol` branch) with VITRIOL
> modifications. The upstream repository's AI-contribution policy (see
> `CONTRIBUTING.md` upstream) applies **only** to PRs against `ggml-org/llama.cpp`.
> In this private fork the VITRIOL project's own rules govern.

## VITRIOL workflow

VITRIOL is built with heavy AI-assistance at the maintainer's direction. The AI agent
(and the maintainer together) write code, commit messages, and reports. Do not defer work
to the maintainer for the sake of "human authorship" — the maintainer has explicitly chosen
this workflow. Commit messages, PR descriptions, and code review responses written by the
agent are welcome.

- The agent writes commits **only when the maintainer explicitly asks** ("commit ...").
- Stage only intended files; never stage unrelated working-tree changes or secrets.
- Write commit messages in normal prose (Conventional Commits, terse, no fluff).
- Follow the VITRIOL project rules in `/home/randozart/Desktop/Projects/VITRIOL/AGENTS.md`:
  testing protocol, documentation reports, code conventions, `VITRIOL_` env vars, licensing.

## Fork specifics

- The VITRIOL predictor lives in `ggml/src/ggml-cuda/vitriol-cuda-integration.cpp`.
- The server context checkpoint logic is in `tools/server/server-context.cpp`.
- All VITRIOL env vars are prefixed with `VITRIOL_` (plus `GGML_CUDA_GDN_PROFILE` for the
  env-gated delta-net/decode timing added 2026-08-19).
- Dual-GPU build requires both archs:
  `cmake -B build -DCMAKE_CUDA_ARCHITECTURES="61;86"` (RTX 3060 sm_86 + GTX 1070 Ti sm_61).
- After any `cmake --build`, the maintainer must run `sudo vitriol setup` before testing
  (sets `CAP_IPC_LOCK` for page-locked DMA buffers).
- Kill stale servers with `killall -9 llama-server` before starting a new one.

## Licensing

This fork is Apache-2.0 (changed from GPL-2.0 on 2026-08-28; the fork is owned by the same
author as VITRIOL). Before incorporating third-party code, check compatibility (copyleft
licenses — GPL/LGPL/AGPL — stay out or get re-derived). Algorithm-bearing modules carry a
`PROVENANCE` header. See root `AGENTS.md` and `docs/provenance/`.

## Useful resources (upstream)

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [Server usage documentation](tools/server/README.md)
- [Server development documentation](tools/server/README-dev.md)
- [How to add a new model](docs/development/HOWTO-add-model.md)
- [Build documentation](docs/build.md)