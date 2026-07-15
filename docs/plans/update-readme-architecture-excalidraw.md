# Update README Architecture and Excalidraw Diagram

## Goal
Refresh the README architecture documentation to match the current Gen-Aperture runtime architecture and add an Excalidraw JSON diagram that can be opened or edited in Excalidraw.

## Non-goals
- Do not change backend or frontend runtime behavior.
- Do not add, remove, or upgrade dependencies.
- Do not rewrite historical task plans.
- Do not read or expose local secret values.

## Constraints
- Treat existing uncommitted code and plan-file changes as user/work changes; do not revert them.
- Use `design.md`, recent task plans, and focused code reads as architecture evidence.
- Keep the README concise enough to remain useful as the project entry point.

## Acceptance criteria
- README and architecture docs reflect current NVIDIA NIM, direct `icc_images_ext` OpenSearch, guarded conversation-store, reranker timeout/model, agent LLM timeout, and UI rerank-progress architecture.
- README no longer relies on stale embedded image references for architecture/workflow where a maintained diagram artifact exists.
- Excalidraw JSON validates as JSON and contains a readable architecture flow.
- Final handoff states commands run, what they proved, and any remaining uncertainty.

## Approach
- Read the current architecture docs and relevant source files for config defaults and request/response shape.
- Update README architecture, agent flow, configuration, project structure, and documentation links where stale.
- Patch small drift in `design.md` and `QUICKSTART.md` where current defaults are duplicated.
- Add `docs/gen-aperture-architecture.excalidraw.json` with grouped system components and labeled arrows.
- Validate JSON parsing and inspect the git diff.

## Files / areas affected
- `README.md`
- `design.md`
- `QUICKSTART.md`
- `docs/gen-aperture-architecture.excalidraw.json`
- `docs/plans/update-readme-architecture-excalidraw.md`

## Verification plan
- Parse the generated Excalidraw JSON with Python's standard `json` module.
- Check README references to the new diagram file and stale image references.
- Review `git diff -- README.md design.md QUICKSTART.md docs/gen-aperture-architecture.excalidraw.json docs/plans/update-readme-architecture-excalidraw.md`.

## Test plan
- Before: README has older architecture details and static image references that are not maintained architecture artifacts.
- Happy path: README points to the generated Excalidraw JSON and describes current runtime flow.
- Sad path: JSON parse failure or broken README reference is caught before handoff.
- After/proof: JSON parse, markdown reference check, and diff review pass.

## Monitoring plan
- Documentation-only change; future architecture changes should update README and the Excalidraw diagram together.

## Risks / open questions
- The exact Excalidraw visual rendering is validated structurally as JSON, not visually inside Excalidraw.
- Existing uncommitted application changes are not owned by this docs update.

## Status
- Implemented and locally verified with JSON parsing, stale-reference scan, diagram reference checks, and diff review.
