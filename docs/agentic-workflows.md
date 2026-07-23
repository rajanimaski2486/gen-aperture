# Gen-Aperture Agentic Workflows

This document describes the agentic search workflows inside Gen-Aperture. It complements the general system diagram in [gen-aperture-architecture.excalidraw.json](gen-aperture-architecture.excalidraw.json) with workflow-focused diagrams and the search patterns each workflow uses.

Editable workflow diagram: [gen-aperture-agentic-workflows.excalidraw.json](gen-aperture-agentic-workflows.excalidraw.json)

## Workflow Summary

Gen-Aperture uses controlled agent graphs rather than an open-ended autonomous agent. The default `agent_squad` workflow is a LangGraph state machine with deterministic routing, specialist agents, direct OpenSearch retrieval, optional reflection reranking, and a final synthesizer. The optional `searchbybrief` workflow is a heavier multi-stage brief search graph with lane planning, candidate recall, precision reranking, and an agentic curation loop.

```mermaid
flowchart TD
    request["POST /api/chat<br/>message + optional PDF/DOCX/TXT"]
    guard{"Write intent?"}
    blocked["Blocked before agent graph<br/>read-only OpenSearch posture"]
    context["Request context<br/>conversation history + stored file context<br/>file text/image extraction"]
    mode{"workflow_mode"}
    squad["Default Agent Squad<br/>router -> optional PM -> search -> synthesize"]
    searchbybrief["Optional SearchByBrief<br/>preprocess -> planner -> retriever -> reranker -> curator loop"]
    response["ChatResponse<br/>results + workflow_steps + rerank log"]
    store["Guarded conversation write"]

    request --> guard
    guard -->|"mutation-like query"| blocked
    guard -->|"read/search query"| context --> mode
    mode -->|"agent_squad"| squad --> response
    mode -->|"searchbybrief"| searchbybrief --> response
    response --> store
```

## Default Agent Squad Workflow

The default path is optimized for conversational stock-photo search. It lets deterministic code make safety and routing decisions, then gives each agent a narrow job.

```mermaid
flowchart TD
    start["AgentSquad.run"]
    router["Squad Router<br/>detect relevance/popular mode<br/>route by file context"]
    file{"File or stored brief context?"}
    pm["Project Manager<br/>extract brand/domain requirements<br/>lexical + semantic queries<br/>filters, exclusions, media type"]
    search["Search Specialist<br/>resolve follow-up context<br/>select image/video/mixed branch<br/>execute retrieval"]
    image[("Direct image retrieval<br/>OpenSearch icc_images_ext<br/>embedding kNN + lexical multi_match")]
    video["Video Search Service path<br/>explicit video/mixed requests"]
    candidates["Candidate results<br/>normalized PhotoResult shape"]
    rerank{"Rerank trigger phrase?"}
    reflect["Reflection Reranker<br/>score -> critique -> deterministic select"]
    synth["Synthesizer<br/>brief analysis + result summary"]
    trace["workflow_steps<br/>reasoning, input/output, OpenSearch payload"]

    start --> router --> file
    file -->|"yes"| pm --> search
    file -->|"no"| search
    search -->|"image"| image --> candidates
    search -->|"video/mixed"| video --> candidates
    candidates --> rerank
    rerank -->|"yes"| reflect --> synth
    rerank -->|"no"| synth
    router -.-> trace
    pm -.-> trace
    search -.-> trace
    reflect -.-> trace
    synth -.-> trace
```

### Agentic Search Patterns

| Pattern | How Gen-Aperture applies it |
| --- | --- |
| Guard before graph | Write-like queries are blocked before LangGraph runs, keeping the production OpenSearch domain read-only. |
| Deterministic router | The Squad Router uses code-driven routing for file/no-file paths and `relevance` versus `popular` search mode, avoiding an LLM call for basic dispatch. |
| Stateful multi-agent handoff | `AgentState` carries conversation context, extracted file content, requirements, query strings, filters, results, rerank output, and workflow trace between graph nodes. |
| Plan then retrieve | Uploaded briefs go to the Project Manager first; retrieval starts only after the brief is converted into concrete lexical/semantic queries, exclusions, and filters. |
| Hybrid recall | Image search combines semantic query embeddings with lexical `multi_match` against `icc_images_ext`, then maps hits into the shared frontend result shape. |
| Reflection after retrieval | Reflection reranking is a post-retrieval judge pattern, triggered only by phrases such as `best`, `top ranked`, `rerank`, `reflect and respond`, or `reviewed`. |
| Bounded fallback | Agent LLM calls and reranker calls have explicit timeouts. Reranker timeout/error returns original candidates with a note instead of failing the whole chat response. |
| Traceable agent work | Each agent appends `workflow_steps` so the UI can show reasoning, decisions, inputs, outputs, and generated OpenSearch payloads. |

## Text-Only Search Pattern

Text-only search is the shortest Agent Squad path. It uses conversation context to resolve follow-ups, prepares query terms, chooses the media branch, and searches directly.

```mermaid
flowchart LR
    query["User query"]
    history["Conversation history<br/>optional follow-up resolution"]
    intent["Search intent<br/>relevance or popular"]
    terms["Query preparation<br/>entities, lexical terms,<br/>semantic query, generated intent"]
    branch{"Media branch"}
    image["Image branch<br/>direct hybrid OpenSearch"]
    mixed["Video or mixed branch<br/>video service + optional image search"]
    results["Top candidates"]
    optional{"Rerank requested?"}
    answer["Synthesized answer"]

    query --> history --> intent --> terms --> branch
    branch -->|"image"| image --> results
    branch -->|"video/mixed"| mixed --> results
    results --> optional
    optional -->|"yes"| answer
    optional -->|"no"| answer
```

Pattern notes:

- The agent is not asking a search service to invent an OpenSearch payload for image search. `PhotoSearchService` builds the image query directly.
- The workflow keeps generated payloads visible in the trace so search behavior can be debugged from the UI.
- Reranking words are metadata about selection quality, not search subject terms.

## Brief-Assisted Search Pattern

When a file is uploaded or stored on the conversation, the Project Manager acts as a brief interpreter before retrieval.

```mermaid
flowchart TD
    brief["Uploaded or stored brief<br/>text + extracted images"]
    vision["Image analyzer<br/>objects, scenes, text, style cues"]
    pm["Project Manager<br/>domain + requirements analysis"]
    outputs["Structured handoff<br/>lexical_query<br/>semantic_query<br/>boolean_query<br/>exclusions, categories, filters"]
    readiness{"Brief searchable?"}
    search["Search Specialist<br/>brief image/video/mixed retrieval"]
    candidates["Candidate assets"]
    rerank{"Optional reflection rerank"}
    synth["Synthesizer<br/>brief analysis + search summary"]

    brief --> vision --> pm
    brief --> pm
    pm --> outputs --> readiness
    readiness -->|"searchable or partial"| search --> candidates --> rerank --> synth
    readiness -->|"weak/insufficient"| search
```

Pattern notes:

- This is a decomposition pattern: the brief is split into subject queries, visual requirements, exclusions, named entities, categories, and media intent.
- The Project Manager explicitly separates searchable subjects from style/quality requirements so retrieval is anchored on concrete image subjects.
- Brief warnings can travel to the synthesizer while still allowing a partial search when there is enough signal.

## Reflection Reranker Pattern

The Reflection Reranker is a controlled reflection pattern over already-retrieved candidates.

```mermaid
flowchart TD
    trigger{"Trigger phrase present?"}
    skip["Skip reranker<br/>return original candidate order"]
    score["Pass 1<br/>LLM scores candidates<br/>using approved ICC fields"]
    critique["Pass 2<br/>LLM critiques high-score candidates"]
    select["Pass 3<br/>deterministic Python selection<br/>dedupe, threshold, rank"]
    timeout{"Timeout or error?"}
    fallback["Fallback<br/>original candidates + reranking note"]
    final["Final selected candidates<br/>rerank decisions + explanation"]

    trigger -->|"no"| skip
    trigger -->|"yes"| score --> critique --> select --> timeout
    timeout -->|"yes"| fallback
    timeout -->|"no"| final
```

Pattern notes:

- The LLM judges only the evidence fields available from `icc_images_ext`; the final keep/discard step is deterministic.
- The reranker is intentionally post-retrieval. It improves precision without changing the recall query itself.
- The timeout cap protects the chat endpoint from becoming an unbounded agent loop.

## Optional SearchByBrief Workflow

`workflow_mode=searchbybrief` uses a separate LangGraph path for heavier brief-driven image selection when optional ML dependencies are installed.

```mermaid
flowchart TD
    preprocess["Brief preprocess<br/>reuse extracted text/images"]
    planner["Stage 0 Planner<br/>intent extraction<br/>multiple search lanes"]
    retriever["Stage 1 Retriever<br/>per-lane candidate recall"]
    reranker["Stage 2 Reranker<br/>precision scoring"]
    curator["Stage 3 Curator<br/>visual scoring, shortlist,<br/>diversity audit"]
    decision{"Curator feedback"}
    repair["Repair directives<br/>refine lanes"]
    final["Final collection"]
    cap["Iteration cap<br/>promote shortlist if needed"]

    preprocess --> planner --> retriever --> reranker --> curator --> decision
    decision -->|"done"| final
    decision -->|"continue"| repair --> planner
    decision -->|"iterations > 3"| cap --> final
```

Pattern notes:

- This path uses lane decomposition: one brief can become many visual search lanes.
- Retrieval is recall-first, then later stages improve precision and collection quality.
- The curator is the agentic loop controller. It can audit missing attributes or duplicate lanes and send repair directives back to the planner.
- The graph has an iteration guard, so repair is bounded and always returns a final collection or promoted shortlist.

## Pattern-To-Code Map

| Workflow area | Main source files |
| --- | --- |
| Chat request, context loading, workflow mode dispatch | `backend/app/routers/chat.py` |
| Agent Squad LangGraph nodes and state | `backend/app/services/agent_squad.py` |
| Direct OpenSearch image retrieval | `backend/app/services/photo_search.py` |
| Reflection reranking | `backend/app/services/reranker.py` |
| Query intent and refinement helpers | `backend/app/services/query_intent.py`, `backend/app/services/query_refinement.py` |
| Optional SearchByBrief graph | `backend/app/services/searchbybrief/main.py` |
| SearchByBrief stage notes | `backend/app/services/searchbybrief/stages.md` |
| Frontend workflow trace and rerank log | `frontend/src/App.jsx` |
