# NVIDIA NIM APIs and RAG Document

## Goal

Create a dedicated document with an editable Excalidraw block diagram explaining how Gen-Aperture uses NVIDIA APIs, how reasoning and embedding calls fit into RAG-style search, how to query those APIs, and how NVIDIA NIM handles model deployment/download concerns.

## Non-goals

- Do not change runtime behavior.
- Do not run package installs, Docker pulls, model downloads, or OpenSearch backfills.
- Do not replace existing architecture or agentic workflow diagrams.
- Do not expose or inspect local secrets.

## Constraints

- Use official NVIDIA documentation for current NIM/API deployment guidance.
- Keep the document specific to this repository's current configuration.
- Make all model download/deployment commands examples only, with placeholder secrets.
- Preserve the current default: OpenAI query embeddings are used until a matching NVIDIA vector field is backfilled.

## Acceptance criteria

- A new Markdown document explains NVIDIA API usage in this application.
- A new editable Excalidraw JSON block diagram shows NVIDIA-hosted and self-hosted NIM paths.
- The document includes steps for NIM credentials, model/container download, reasoning API calls, embedding API calls, and RAG/query flow.
- README links to the new document and diagram.

## Approach

- Read app config and NVIDIA-related services to map actual code paths.
- Verify NIM LLM, embedding, RAG, and deployment details against official NVIDIA docs.
- Add `docs/nvidia-nim-apis-rag.md` with Mermaid diagrams, command examples, and source links.
- Add `docs/nvidia-nim-apis-rag.excalidraw.json` as an editable block diagram.
- Update README documentation links.

## Files / areas affected

- `README.md`
- `docs/nvidia-nim-apis-rag.md`
- `docs/nvidia-nim-apis-rag.excalidraw.json`
- `docs/plans/nvidia-nim-apis-rag-document.md`

## Verification plan

- Parse the new Excalidraw JSON with Python's standard `json` module.
- Check local Markdown links resolve.
- Run `git diff --check`.
- Inspect `git status --short` to confirm changes are documentation-only.

## Test plan

- Before: confirm the repo lacks a NVIDIA/NIM/RAG-specific document.
- Happy path: the new doc explains both hosted NVIDIA API Catalog usage and self-hosted NIM usage.
- Sad path: call out that model download/deployment commands are examples only and were not executed.
- After/proof: JSON parse, local-link checks, whitespace checks, and diff/status inspection pass.

## Monitoring plan

- Documentation-only change. Update this document when model names, NIM endpoints, embedding dimensions, or vector-field migration steps change.

## Risks / open questions

- NVIDIA model availability and exact NIM container tags change over time; readers must consult the current NVIDIA support matrix/API catalog before running deployment commands.
- The repo's current query-time embedding default remains OpenAI until NVIDIA vectors are backfilled.

## Status

- Implemented and locally verified with Excalidraw JSON parsing, local Markdown link checks, whitespace checks, diagram-label inspection, and diff/status inspection. This is a documentation-only change; no Docker pulls, model downloads, installs, or OpenSearch backfills were run.
