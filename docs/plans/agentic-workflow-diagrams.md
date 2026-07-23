# Agentic Workflow Diagrams

## Goal

Add dedicated documentation and editable diagrams for the agentic search workflows in Gen-Aperture, separate from the existing general application architecture diagram.

## Non-goals

- Do not change backend, frontend, or runtime behavior.
- Do not replace the existing general architecture Excalidraw diagram.
- Do not introduce new documentation tooling or generated-site dependencies.

## Constraints

- Keep the diagrams editable as Excalidraw JSON.
- Base the workflow descriptions on current repo code and docs.
- Keep documentation readable in GitHub Markdown even without opening Excalidraw.
- Avoid package installation or lockfile changes.

## Acceptance criteria

- A dedicated agentic workflow document explains the application's agentic search patterns.
- The document includes workflow diagrams for the default Agent Squad path, the brief-assisted path, the reflection reranker, and the optional SearchByBrief loop.
- A new Excalidraw JSON artifact focuses on the agentic workflow rather than general system architecture.
- README/design documentation links distinguish the general architecture diagram from the agentic workflow diagrams.

## Approach

- Read current architecture docs and agent orchestration code for factual workflow evidence.
- Add `docs/agentic-workflows.md` with Mermaid diagrams and concise pattern explanations.
- Add `docs/gen-aperture-agentic-workflows.excalidraw.json` as an editable workflow-focused diagram.
- Update README and design documentation links to point to the new workflow doc and diagram.

## Files / areas affected

- `README.md`
- `design.md`
- `docs/agentic-workflows.md`
- `docs/gen-aperture-agentic-workflows.excalidraw.json`
- `docs/plans/agentic-workflow-diagrams.md`

## Verification plan

- Parse all Excalidraw JSON files with Python's standard `json` module.
- Confirm README/design links point at existing files.
- Inspect the diff to ensure changes are documentation-only and match the requested scope.

## Test plan

- Before: confirm only the general architecture Excalidraw exists.
- Happy path: Markdown doc renders with Mermaid source blocks and links to both editable diagrams.
- Sad path: if Excalidraw rendering is unavailable, the Mermaid diagrams and prose still describe each workflow.
- After/proof: JSON parse and link-existence checks pass; diff shows no runtime files changed.

## Monitoring plan

- Documentation-only change. Future agent workflow changes should update `docs/agentic-workflows.md` and the focused Excalidraw diagram alongside code changes.

## Risks / open questions

- The optional SearchByBrief path depends on heavy ML dependencies and may not be installed locally; this plan documents its current graph and stages from source rather than executing it.
- Excalidraw visual layout is structurally validated as JSON, not rendered inside Excalidraw in this task.

## Status

- Implemented and locally verified with Excalidraw JSON parsing, local Markdown link checks, whitespace checks, and diff/status inspection. This is a documentation-only change.
