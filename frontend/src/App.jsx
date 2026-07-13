import { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./index.css";
import { chatAPI, conversationsAPI } from "./services/api";

const ACTIVE_CONVERSATION_KEY = "active_conversation_id";
const MAX_RESULTS_DISPLAYED = 24;

// ---------------------------------------------------------------------------
// Demo Mode configuration
// ---------------------------------------------------------------------------
const DEMO_STEPS = [
  {
    label: '📋 Upload Brief',
    message: 'Analyze this Raspberries marketing campaign brief and find matching stock photos',
    attachBrief: true,
  },
  {
    label: '🔧 Refine Results',
    message: 'Show me horizontal orientation images only',
    attachBrief: false,
  },
  {
    label: '🔥 Popular Mode',
    message: 'Show me the most popular summer beverage images',
    attachBrief: false,
  },
  {
    label: '🎯 Rerank',
    message: 'best results — reflect and rerank',
    attachBrief: false,
  },
];

/** Modal to display JSON payload */
function PayloadModal({ title, payload, url, method, onClose }) {
  if (!payload) return null;
  return (
    <div className="payload-modal-overlay" onClick={onClose}>
      <div className="payload-modal" onClick={(e) => e.stopPropagation()}>
        <div className="payload-modal-header">
          <h3>{title}</h3>
          <button className="payload-modal-close" onClick={onClose}>
            ✕
          </button>
        </div>
        {url && (
          <div className="payload-modal-url">
            <span className="payload-modal-url-label">{method || "POST"}</span>
            <code>{url}</code>
          </div>
        )}
        <pre className="payload-modal-body">
          {JSON.stringify(payload, null, 2)}
        </pre>
      </div>
    </div>
  );
}

/** Collapsible Brief Analysis section inside an assistant message */
function BriefAnalysisSection({ content }) {
  const [expanded, setExpanded] = useState(true);

  const BRIEF_MARKER = '**Brief Analysis:**';
  const briefStart = content.indexOf(BRIEF_MARKER);

  if (briefStart === -1) {
    return (
      <div className="markdown-body">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
      </div>
    );
  }

  const before = content.slice(0, briefStart).trim();
  // Find where the next top-level section begins (e.g. "**Search Results")
  const afterBriefRaw = content.slice(briefStart + BRIEF_MARKER.length);
  const nextSectionMatch = afterBriefRaw.match(/\n\n\*\*(?!Structured Analysis)/);
  const splitIdx = nextSectionMatch ? nextSectionMatch.index : afterBriefRaw.length;
  const briefContent = afterBriefRaw.slice(0, splitIdx).trim();
  const after = afterBriefRaw.slice(splitIdx).trim();

  return (
    <div className="markdown-body">
      {before && <ReactMarkdown remarkPlugins={[remarkGfm]}>{before}</ReactMarkdown>}

      <div className="brief-analysis-panel">
        <button
          className="brief-analysis-toggle"
          onClick={() => setExpanded(prev => !prev)}
        >
          <span>📋 Brief Analysis</span>
          <span style={{ marginLeft: 'auto' }}>{expanded ? '▾' : '▸'}</span>
        </button>
        {expanded && (
          <div className="brief-analysis-content">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{briefContent}</ReactMarkdown>
          </div>
        )}
      </div>

      {after && <ReactMarkdown remarkPlugins={[remarkGfm]}>{after}</ReactMarkdown>}
    </div>
  );
}

/** Reflection Reranker log panel */
function RerankLogPanel({ decisions, explanation }) {
  const [expanded, setExpanded] = useState(false);

  if (!decisions || decisions.length === 0) return null;

  const kept = decisions.filter((d) => d.keep);
  const discarded = decisions.filter((d) => !d.keep);
  const total = decisions.length;

  return (
    <div className="rerank-log-panel">
      <button
        className="rerank-log-toggle"
        onClick={() => setExpanded((prev) => !prev)}
      >
        <span>🎯 Reflection Reranking Log</span>
        <span className="rerank-log-summary">
          {total} evaluated → {kept.length} selected
        </span>
        <span style={{ marginLeft: "auto" }}>{expanded ? "▾" : "▸"}</span>
      </button>

      {expanded && (
        <div className="rerank-log-body">
          {explanation && (
            <div className="rerank-explanation-box">⚠️ {explanation}</div>
          )}

          <table className="rerank-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Image</th>
                <th>Score</th>
                <th>Decision</th>
                <th>Reason</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {/* Kept results first */}
              {kept
                .sort((a, b) => (a.final_rank ?? 999) - (b.final_rank ?? 999))
                .map((d, i) => (
                  <tr
                    key={`keep-${i}`}
                    className={d.is_borderline ? "row-borderline" : "row-keep"}
                  >
                    <td>#{d.final_rank}</td>
                    <td className="cell-description">{d.hadron_id || "—"}</td>
                    <td>{d.rerank_score?.toFixed(2)}</td>
                    <td>
                      {d.is_borderline ? (
                        <span className="decision-borderline">
                          ⚠ Borderline
                        </span>
                      ) : (
                        <span className="decision-keep">✓ Keep</span>
                      )}
                    </td>
                    <td className="cell-reason">{d.reason || "—"}</td>
                    <td>
                      {d.confidence != null
                        ? `${(d.confidence * 100).toFixed(0)}%`
                        : "—"}
                    </td>
                  </tr>
                ))}
              {/* Discarded results */}
              {discarded.map((d, i) => (
                <tr key={`discard-${i}`} className="row-discard">
                  <td>—</td>
                  <td className="cell-description">{d.hadron_id || "—"}</td>
                  <td>{d.rerank_score?.toFixed(2)}</td>
                  <td>
                    <span className="decision-discard">✗ Discard</span>
                  </td>
                  <td className="cell-reason">{d.reason || "—"}</td>
                  <td>
                    {d.confidence != null
                      ? `${(d.confidence * 100).toFixed(0)}%`
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/** Agent workflow visualization panel */
function AgentWorkflowPanel({ steps }) {
  const [expanded, setExpanded] = useState(false);
  const [expandedSteps, setExpandedSteps] = useState({});
  const [payloadModal, setPayloadModal] = useState(null);

  const toggleStep = (idx) => {
    setExpandedSteps((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  const agentIcons = {
    "Squad Router": "🧭",
    "Project Manager": "📋",
    "Search Specialist": "🔍",
    Synthesizer: "✨",
    "Reflection Reranker": "🎯",
  };

  const agentColors = {
    "Squad Router": "#6366f1",
    "Project Manager": "#f59e0b",
    "Search Specialist": "#10b981",
    Synthesizer: "#8b5cf6",
    "Reflection Reranker": "#0d9488",
  };

  return (
    <div className="workflow-panel">
      {payloadModal && (
        <PayloadModal
          title={payloadModal.title}
          payload={payloadModal.payload}
          url={payloadModal.url}
          method={payloadModal.method}
          onClose={() => setPayloadModal(null)}
        />
      )}
      <button
        className="workflow-toggle"
        onClick={() => setExpanded(!expanded)}
      >
        <span className="workflow-toggle-icon">{expanded ? "▾" : "▸"}</span>
        <span>🤖 Agent Workflow</span>
        <span className="workflow-badge">{steps.length} steps</span>
      </button>

      {expanded && (
        <div className="workflow-content">
          {/* Flow diagram */}
          <div className="workflow-flow">
            {steps.map((step, idx) => (
              <div key={idx} className="workflow-flow-item">
                <div
                  className="workflow-flow-node"
                  style={{ borderColor: agentColors[step.agent] || "#6b7280" }}
                >
                  <span>{agentIcons[step.agent] || "⚙️"}</span>
                  <span>{step.agent}</span>
                </div>
                {idx < steps.length - 1 && (
                  <div className="workflow-flow-arrow">→</div>
                )}
              </div>
            ))}
          </div>

          {/* Detailed steps */}
          <div className="workflow-steps">
            {steps.map((step, idx) => (
              <div key={idx} className="workflow-step">
                <div
                  className="workflow-step-header"
                  onClick={() => toggleStep(idx)}
                >
                  <div className="workflow-step-title">
                    <span
                      className="workflow-step-dot"
                      style={{
                        background: agentColors[step.agent] || "#6b7280",
                      }}
                    />
                    <span className="workflow-step-agent">
                      {agentIcons[step.agent] || "⚙️"} {step.agent}
                    </span>
                    <span className="workflow-step-action">
                      — {step.action}
                    </span>
                    {step.model && (
                      <span className="workflow-step-model" title={`Model: ${step.model}`}>
                        🤖 {step.model}
                      </span>
                    )}
                  </div>
                  <span className="workflow-step-expand">
                    {expandedSteps[idx] ? "▾" : "▸"}
                  </span>
                </div>

                <div className="workflow-step-reasoning">
                  💭 {step.reasoning}
                  {step.opensearch_payload && (
                    <button
                      className="payload-link"
                      onClick={(e) => {
                        e.stopPropagation();
                        setPayloadModal({
                          title: `${step.agent} — OpenSearch Payload`,
                          payload: step.opensearch_payload,
                          url: step.opensearch_url,
                        });
                      }}
                    >
                      📋 View OpenSearch Payload
                    </button>
                  )}
                  {step.search_service_response && (
                    <button
                      className="payload-link"
                      onClick={(e) => {
                        e.stopPropagation();
                        setPayloadModal({
                          title: `Search Service Response — ${step.action}`,
                          payload: step.search_service_response,
                          url: step.search_service_endpoint,
                          method: "GET",
                        });
                      }}
                    >
                      🌐 View Search Service Response
                    </button>
                  )}
                </div>

                {expandedSteps[idx] && (
                  <div className="workflow-step-details">
                    {step.decision && (
                      <div className="workflow-detail-block">
                        <div className="workflow-detail-label">
                          🔀 Route Decision
                        </div>
                        <code>{step.decision}</code>
                      </div>
                    )}
                    {step.prompt && (
                      <div className="workflow-detail-block">
                        <div className="workflow-detail-label">
                          📝 System Prompt
                        </div>
                        <pre className="workflow-prompt">{step.prompt}</pre>
                      </div>
                    )}
                    {step.input && (
                      <div className="workflow-detail-block">
                        <div className="workflow-detail-label">📥 Input</div>
                        <pre className="workflow-json">
                          {JSON.stringify(step.input, null, 2)}
                        </pre>
                      </div>
                    )}
                    {step.output && (
                      <div className="workflow-detail-block">
                        <div className="workflow-detail-label">📤 Output</div>
                        <pre className="workflow-json">
                          {JSON.stringify(step.output, null, 2)}
                        </pre>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function App() {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState("");
  const [conversationId, setConversationId] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  // True when the outgoing message contains a rerank trigger phrase
  const [isReranking, setIsReranking] = useState(false);
  const [error, setError] = useState(null);
  const [workflowMode, setWorkflowMode] = useState('agent_squad');
  // Model selection
  const [selectedModel, setSelectedModel] = useState('meta/llama-3.3-70b-instruct');
  const [showModelSelector, setShowModelSelector] = useState(false);
  const availableModels = [
    { id: 'meta/llama-3.3-70b-instruct', name: 'Llama 3.3 70B', description: 'NVIDIA hosted reasoning model' },
    { id: 'meta/llama-3.1-70b-instruct', name: 'Llama 3.1 70B', description: 'NVIDIA hosted fallback option' },
  ];
  // Tracks the index of the video card currently being hovered (for hover-play)
  const [hoveredCard, setHoveredCard] = useState(null);

  // Demo mode
  const [isDemoMode, setIsDemoMode] = useState(false);
  const [demoStep, setDemoStep] = useState(0);

  // Initialize app state on mount
  useEffect(() => {
    const initializeApp = async () => {
      sessionStorage.removeItem("openai_api_key");

      // Load stored model preference
      const storedModel = sessionStorage.getItem('selected_model');
      if (storedModel && availableModels.some((model) => model.id === storedModel)) {
        setSelectedModel(storedModel);
      } else {
        sessionStorage.setItem('selected_model', selectedModel);
      }

      await loadRecentConversations();

      // Restore the last active conversation after page reload.
      // Use silent mode so a missing conversation (e.g. after backend restart)
      // doesn't surface as an error toast.
      const storedConversationId = sessionStorage.getItem(
        ACTIVE_CONVERSATION_KEY,
      );
      if (storedConversationId) {
        const loaded = await loadConversation(storedConversationId, {
          silent: true,
        });
        if (!loaded) {
          sessionStorage.removeItem(ACTIVE_CONVERSATION_KEY);
        }
      }
    };

    initializeApp();
  }, []);

  const loadRecentConversations = async () => {
    try {
      const recent = await conversationsAPI.getRecent();
      setConversations(recent);
      return recent;
    } catch (err) {
      console.error("Failed to load conversations:", err);
      return [];
    }
  };

  const loadConversation = async (convId, { silent = false } = {}) => {
    try {
      setIsLoading(true);
      const conversation = await conversationsAPI.getConversation(convId);

      // Transform backend messages to frontend format
      const loadedMessages = [];
      if (conversation.messages && conversation.messages.length > 0) {
        conversation.messages.forEach((msg) => {
          // Add user message
          loadedMessages.push({
            role: "user",
            content: msg.user_message,
            file: conversation.file_name || null,
          });
          // Add assistant message
          loadedMessages.push({
            role: "assistant",
            content: msg.agent_response,
            results: msg.search_results_count || null,
          });
        });
      }

      setMessages(loadedMessages);
      setConversationId(convId);
      sessionStorage.setItem(ACTIVE_CONVERSATION_KEY, convId);
      setSelectedFile(null);
      return true;
    } catch (err) {
      console.error("Failed to load conversation:", err);
      // Don't surface a 404 as an error — the conversation simply no longer
      // exists (e.g. backend was restarted and in-memory history was cleared).
      // Only show an error toast for unexpected failures.
      const isNotFound = err?.response?.status === 404;
      if (!silent && !isNotFound) {
        showError("Failed to load conversation history");
      }
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  // Close model selector when clicking outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      const modelSelector = document.querySelector('.model-selector-container');
      if (modelSelector && !modelSelector.contains(e.target)) {
        setShowModelSelector(false);
      }
    };

    if (showModelSelector) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [showModelSelector]);

  const handleNewChat = () => {
    setConversationId(null);
    sessionStorage.removeItem(ACTIVE_CONVERSATION_KEY);
    setMessages([]);
    setSelectedFile(null);
  };

  // ---------------------------------------------------------------------------
  // Demo Mode handlers
  // ---------------------------------------------------------------------------
  const handleStartDemo = () => {
    setSelectedModel('meta/llama-3.3-70b-instruct');
    sessionStorage.setItem('selected_model', 'meta/llama-3.3-70b-instruct');
    setConversationId(null);
    sessionStorage.removeItem(ACTIVE_CONVERSATION_KEY);
    setMessages([]);
    setSelectedFile(null);
    setIsLoading(false);
    setDemoStep(0);
    setIsDemoMode(true);
  };

  const handleEndDemo = () => {
    setIsDemoMode(false);
    setDemoStep(0);
  };

  const handleDemoStep = async (stepIdx) => {
    const step = DEMO_STEPS[stepIdx];
    setDemoStep(stepIdx + 1);

    try {
      let file = null;
      if (step.attachBrief) {
        const resp = await fetch('/demo-brief.pdf');
        if (!resp.ok) throw new Error(`Could not load demo brief (${resp.status})`);
        const blob = await resp.blob();
        file = new File([blob], 'RASPBERRIES Marketing Campaign.pdf', { type: 'application/pdf' });
      }
      await handleSendMessage(step.message, file);
    } catch (err) {
      console.error('Demo step error:', err);
      showError(`Demo step failed: ${err.message}`);
      setDemoStep(stepIdx);  // revert so the step is retryable
    }
  };

  const handleModelSelect = (modelId) => {
    setSelectedModel(modelId);
    sessionStorage.setItem('selected_model', modelId);
    setShowModelSelector(false);
  };

  const handleDeleteConversation = async (e, convId) => {
    e.stopPropagation();
    try {
      await conversationsAPI.deleteConversation(convId);
      // If the deleted conversation is currently open, clear it
      if (convId === conversationId) {
        setConversationId(null);
        sessionStorage.removeItem(ACTIVE_CONVERSATION_KEY);
        setMessages([]);
        setSelectedFile(null);
      }
      await loadRecentConversations();
    } catch (err) {
      console.error("Failed to delete conversation:", err);
      showError("Failed to delete conversation");
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (file.size > 6 * 1024 * 1024) {
        showError("File size exceeds 6MB limit");
        return;
      }
      setSelectedFile(file);
    }
  };

  const handleSendMessage = async (directMessage, directFile) => {
    const userMessage = (directMessage !== undefined ? directMessage : inputMessage).trim();
    const fileToSend = directFile !== undefined ? directFile : selectedFile;

    if (!userMessage && !fileToSend) return;

    if (directMessage === undefined) setInputMessage('');

    // Detect rerank trigger phrases to show the dedicated loading indicator
    const rerankTrigger =
      /\bbest\b|\btop[\s-]?ranked?\b|\brerank\b|\breflect\s+and\s+respond\b|\breviewed\b/i;
    setIsReranking(rerankTrigger.test(userMessage));

    // Add user message to UI
    const newUserMessage = {
      role: "user",
      content: userMessage,
      file: fileToSend?.name,
    };
    setMessages((prev) => [...prev, newUserMessage]);

    setIsLoading(true);
  const requestStart = Date.now();

  try {
      const response = await chatAPI.sendMessage(
        userMessage,
        conversationId,
        fileToSend,
        selectedModel,
        workflowMode,
      );
  const generationMs = Date.now() - requestStart;

      // Add agent response
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: response.response,
          results: response.results,
          filter_metadata: response.filter_metadata || null,
          pdf_search_detail: response.pdf_search_detail || null,
          workflow_steps: response.workflow_steps || [],
          search_mode: response.search_mode || 'relevance',
          rerank_applied: response.rerank_applied || false,
          rerank_decisions: response.rerank_decisions || [],
          rerank_explanation: response.rerank_explanation || null,
          generation_ms: generationMs,
        },
      ]);

      // Update conversation ID if new
      if (!conversationId) {
        setConversationId(response.conversation_id);
        sessionStorage.setItem(
          ACTIVE_CONVERSATION_KEY,
          response.conversation_id,
        );
      }

      // Clear file selection
      setSelectedFile(null);

      // Refresh sidebar in background so chat input unlocks immediately.
      loadRecentConversations().catch((err) => {
        console.error('Failed to refresh conversations:', err);
      });

    } catch (err) {
      console.error("Chat error:", err);
      console.error("Error status:", err.response?.status);
      console.error("Error detail:", err.response?.data?.detail);

      // Remove the user message that was just added since it failed
      setMessages((prev) => prev.slice(0, -1));

      if (err.response?.status === 401) {
        const errorMsg =
          err.response?.data?.detail ||
          "Authentication failed. Check the server NVIDIA_API_KEY setting.";
        showError(errorMsg);
      } else {
        showError(err.response?.data?.detail || "Failed to send message");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const showError = (message) => {
    setError(message);
    setTimeout(() => setError(null), 5000);
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const buildImageVariantUrl = (extId, size = '600w') => {
    if (!extId) return '';
    return `http://localhost:9200/assets/image-${size}-${extId}.jpg`;
  };

  const getImageVariants = (result) => {
    const extId = result?.ext_id;
    if (!extId) {
      return {
        src: result?.image_url || result?.thumbnail_url || '',
        srcSet: '',
        previewUrl: result?.image_url || result?.thumbnail_url || '',
      };
    }

    const small = buildImageVariantUrl(extId, '250nw');
    const medium = buildImageVariantUrl(extId, '600w');
    const large = buildImageVariantUrl(extId, '1000w');

    return {
      src: medium,
      srcSet: `${small} 250w, ${medium} 600w, ${large} 1000w`,
      previewUrl: large,
    };
  };

  return (
    <div className="app">
      {/* Sidebar */}
      <div className="sidebar">
        <h2>Recent Conversations</h2>
        <div className="conversation-list">
          {conversations.map((conv) => (
            <div
              key={conv.conversation_id}
              className={`conversation-item ${conv.conversation_id === conversationId ? "active" : ""}`}
              onClick={() => loadConversation(conv.conversation_id)}
            >
              <div className="conversation-item-content">
                <div className="conversation-query">
                  {conv.title || conv.last_user_query || "New conversation"}
                </div>
                <div className="conversation-meta">
                  {conv.message_count} message
                  {conv.message_count !== 1 ? "s" : ""}
                </div>
              </div>
              <button
                className="conversation-delete-btn"
                title="Delete conversation"
                onClick={(e) =>
                  handleDeleteConversation(e, conv.conversation_id)
                }
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Main chat */}
      <div className="chat-container">
        <div className="header">
          <h1>Gen-Aperture</h1>
          
          {/* Model Selector */}
          <div className="model-selector-container">
            <button 
              className="model-selector-btn"
              onClick={() => setShowModelSelector(!showModelSelector)}
              title="Select AI model"
            >
              🤖 {availableModels.find(m => m.id === selectedModel)?.name || 'Model'}
              <span className="model-selector-icon">{showModelSelector ? '▾' : '▸'}</span>
            </button>
            
            {showModelSelector && (
              <div className="model-selector-dropdown">
                {availableModels.map(model => (
                  <div
                    key={model.id}
                    className={`model-option ${model.id === selectedModel ? 'active' : ''}`}
                    onClick={() => handleModelSelect(model.id)}
                  >
                    <div className="model-option-name">{model.name}</div>
                    <div className="model-option-desc">{model.description}</div>
                    {model.id === selectedModel && <span className="model-checkmark">✓</span>}
                  </div>
                ))}
              </div>
            )}
          </div>

          <button className="new-chat-btn" onClick={handleNewChat}>
            + New Chat
          </button>
          <button className="demo-btn" onClick={handleStartDemo}>
            ▶ Demo
          </button>
        </div>

        {/* Demo Control Strip */}
        {isDemoMode && (
          <div className="demo-control-strip">
            <span className="demo-banner">🎬 DEMO MODE</span>
            {DEMO_STEPS.map((step, idx) => (
              <button
                key={idx}
                className={`demo-step-btn ${demoStep > idx ? 'done' : ''} ${demoStep === idx && !isLoading ? 'active' : ''}`}
                onClick={() => handleDemoStep(idx)}
                disabled={isLoading || demoStep !== idx}
              >
                {demoStep > idx ? '✓ ' : ''}{step.label}
              </button>
            ))}
            {isLoading && <span className="demo-sending">⏳ Sending…</span>}
            <button className="demo-end-btn" onClick={handleEndDemo}>
              ✕ End Demo
            </button>
          </div>
        )}

        <div className="messages-area">
          {messages.length === 0 && (
            <div
              style={{ textAlign: "center", color: "#999", marginTop: "4rem" }}
            >
              <h2>Welcome to Gen-Aperture</h2>
              <p>Start a conversation to search for stock photos</p>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              <div className="message-avatar">
                {msg.role === "user" ? "U" : "A"}
              </div>
              <div className="message-content">
                {msg.role === 'assistant' ? (
                  <BriefAnalysisSection content={msg.content} />
                ) : (
                  <div style={{ whiteSpace: "pre-wrap" }}>{msg.content}</div>
                )}
                {msg.role === "assistant" && msg.pdf_search_detail && (
                  <div className="search-detail">
                    <div className="search-detail-title">PDF Search Detail</div>
                    <div>
                      <strong>Images extracted:</strong>{" "}
                      {msg.pdf_search_detail.images_extracted ?? 0}
                    </div>
                    <div>
                      <strong>Text extracted:</strong>{" "}
                      {msg.pdf_search_detail.text_extracted ? "Yes" : "No"}
                    </div>
                    <div>
                      <strong>Enrichment added:</strong>{" "}
                      {msg.pdf_search_detail.enrichment_added?.length > 0
                        ? msg.pdf_search_detail.enrichment_added.join(", ")
                        : "none"}
                    </div>
                    {msg.pdf_search_detail.quality && (
                      <div>
                        <strong>Quality:</strong> {msg.pdf_search_detail.quality}
                      </div>
                    )}
                    {msg.pdf_search_detail.warnings?.length > 0 && (
                      <div>
                        <strong>Warnings:</strong>{" "}
                        {msg.pdf_search_detail.warnings.join(" | ")}
                      </div>
                    )}
                    {msg.pdf_search_detail.gaps?.length > 0 && (
                      <div>
                        <strong>Gaps:</strong>{" "}
                        {msg.pdf_search_detail.gaps.join(" | ")}
                      </div>
                    )}
                  </div>
                )}
                {msg.file && (
                  <div
                    style={{
                      marginTop: "0.5rem",
                      fontSize: "0.875rem",
                      color: "#666",
                    }}
                  >
                    📎 {msg.file}
                  </div>
                )}

                {/* Agent Workflow Panel */}
                {msg.workflow_steps && msg.workflow_steps.length > 0 && (
                  <AgentWorkflowPanel steps={msg.workflow_steps} />
                )}

                {/* Reflection Reranker Log Panel */}
                {msg.rerank_applied && (
                  <RerankLogPanel
                    decisions={msg.rerank_decisions}
                    explanation={msg.rerank_explanation}
                  />
                )}

                {/* Mixed image + video results */}
                {msg.results && msg.results.length > 0 && (() => {
                  const activeResults = msg.results;
                  const imageCount = activeResults.filter(r => r.media_type !== 'video').length;
                  const videoCount = activeResults.filter(r => r.media_type === 'video').length;
                  return (
                    <div className="image-results">
                      <div className="image-results-header">
                        Showing {activeResults.length} results
                        {imageCount > 0 && <span className="result-type-pill result-type-pill--image">📸 {imageCount} images</span>}
                        {videoCount > 0 && <span className="result-type-pill result-type-pill--video">🎬 {videoCount} videos</span>}
                        {msg.search_mode && (
                          <span className={`search-mode-badge ${msg.search_mode}`}>
                            {msg.search_mode === 'popular' ? '🔥 Popular' : '🎯 Relevant'}
                          </span>
                        )}
                        {msg.rerank_applied && (
                          <span className="rerank-badge">🎯 Reranked</span>
                        )}
                        {msg.generation_ms != null && (
                          <span className="generation-time-badge">⏱ {(msg.generation_ms / 1000).toFixed(1)}s</span>
                        )}
                      </div>

                      {msg.filter_metadata?.filters_applied && (
                        <div className="filter-metadata-banner">
                          {msg.filter_metadata.category_values?.length > 0 && (
                            <span>🏷️ {msg.filter_metadata.category_values.join(', ')}</span>
                          )}
                          {msg.filter_metadata.exclusion_terms?.length > 0 && (
                            <span>🚫 Excluded: {msg.filter_metadata.exclusion_terms.join(', ')}</span>
                          )}
                          {msg.filter_metadata.refinement_filter_descriptions?.length > 0 && (
                            <span>🔧 {msg.filter_metadata.refinement_filter_descriptions.join(' · ')}</span>
                          )}
                        </div>
                      )}

                      <div className="image-grid">
                        {activeResults.slice(0, 10).map((result, resultIdx) => {
                          const isVideo = result.media_type === 'video';
                          const cardKey = `${idx}-${resultIdx}`;

                          if (isVideo) {
                            return (
                              <div
                                key={resultIdx}
                                className="image-card image-card--video"
                                onClick={() => window.open(result.video_url, '_blank')}
                              >
                                {result.video_url && (
                                  <video
                                    className="video-card-player"
                                    src={result.video_url}
                                    autoPlay
                                    muted
                                    loop
                                    playsInline
                                  />
                                )}
                                <div className="video-badge">🎬 Video</div>
                                <div className="image-info">
                                  <div className="image-description" title={result.description}>
                                    {result.description}
                                  </div>
                                  <div className="image-meta">
                                    <span>🏆 {result.license_count || 0} licenses</span>
                                    {result.date_added && (
                                      <span>📅 {new Date(result.date_added).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })}</span>
                                    )}
                                    {result.score && (
                                      <span>⭐ {result.score.toFixed(2)}</span>
                                    )}
                                  </div>
                                </div>
                              </div>
                            );
                          }

                          // Image tile
                          return (
                            <div key={resultIdx} className={`image-card${result.is_generated ? ' image-card--ai-generated' : ''}`}>
                              {(() => {
                                const imageVariants = getImageVariants(result);
                                return (
                                  <img
                                    src={imageVariants.src}
                                    srcSet={imageVariants.srcSet || undefined}
                                    sizes="(max-width: 600px) 50vw, (max-width: 900px) 45vw, (max-width: 1200px) 30vw, 22vw"
                                    alt={result.description}
                                    loading="lazy"
                                    onClick={() => window.open(imageVariants.previewUrl, '_blank')}
                                    style={{ cursor: 'pointer' }}
                                  />
                                );
                              })()}
                              {result.is_generated && (
                                <div className="ai-generated-badge">✨ AI Generated</div>
                              )}
                              <div className="image-info">
                                <div className="image-description" title={result.description}>
                                  {result.description}
                                </div>
                                <div className="image-meta">
                                  <span>🏆 {result.license_count || 0} licenses</span>
                                  {result.date_added && (
                                    <span>📅 {new Date(result.date_added).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })}</span>
                                  )}
                                  {result.score && (
                                    <span>⭐ {result.score.toFixed(2)}</span>
                                  )}
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  );
                })()}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="message assistant">
              <div className="message-avatar">A</div>
              <div className="message-content">
                <div className="loading"></div>
                {isReranking && (
                  <p className="rerank-loading-text">
                    🔄 Applying reflection reranking…
                  </p>
                )}
              </div>
            </div>
          )}
        </div>

        <div className="input-area">
          <div className="workflow-mode-toggle">
            <button
              className={`workflow-mode-btn ${workflowMode === 'agent_squad' ? 'active' : ''}`}
              onClick={() => setWorkflowMode('agent_squad')}
              disabled={isLoading}
            >
              Agent Squad
            </button>
            <button
              className={`workflow-mode-btn ${workflowMode === 'searchbybrief' ? 'active' : ''}`}
              onClick={() => setWorkflowMode('searchbybrief')}
              disabled={isLoading}
            >
              SearchByBrief
            </button>
          </div>
          <div className="input-container">
            <label className="file-upload-label" htmlFor="file-upload">
              📎
            </label>
            <input
              id="file-upload"
              type="file"
              className="file-upload-input"
              accept=".pdf,.docx,.txt"
              onChange={handleFileSelect}
            />

            <div className="message-input-wrapper">
              <textarea
                className="message-input"
                placeholder="Type your message..."
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                rows={1}
                disabled={isLoading}
              />
              <button
                className="send-btn"
                onClick={handleSendMessage}
                disabled={isLoading || (!inputMessage.trim() && !selectedFile)}
              >
                {isLoading ? <div className="loading"></div> : "→"}
              </button>
            </div>
          </div>

          {selectedFile && (
            <div className="file-preview">
              <div className="file-preview-info">
                📎 {selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)}{" "}
                KB)
              </div>
              <button
                className="file-preview-remove"
                onClick={() => setSelectedFile(null)}
              >
                ✕
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Error Toast */}
      {error && <div className="toast">{error}</div>}
    </div>
  );
}

export default App;
