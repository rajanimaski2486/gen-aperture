import { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './index.css';
import { chatAPI, conversationsAPI } from './services/api';

const ACTIVE_CONVERSATION_KEY = 'active_conversation_id';

/** Modal to display JSON payload */
function PayloadModal({ title, payload, url, method, onClose }) {
  if (!payload) return null;
  return (
    <div className="payload-modal-overlay" onClick={onClose}>
      <div className="payload-modal" onClick={e => e.stopPropagation()}>
        <div className="payload-modal-header">
          <h3>{title}</h3>
          <button className="payload-modal-close" onClick={onClose}>✕</button>
        </div>
        {url && (
          <div className="payload-modal-url">
            <span className="payload-modal-url-label">{method || 'POST'}</span>
            <code>{url}</code>
          </div>
        )}
        <pre className="payload-modal-body">{JSON.stringify(payload, null, 2)}</pre>
      </div>
    </div>
  );
}

/** Reflection Reranker log panel */
function RerankLogPanel({ decisions, explanation }) {
  const [expanded, setExpanded] = useState(false);

  if (!decisions || decisions.length === 0) return null;

  const kept = decisions.filter(d => d.keep);
  const discarded = decisions.filter(d => !d.keep);
  const total = decisions.length;

  return (
    <div className="rerank-log-panel">
      <button
        className="rerank-log-toggle"
        onClick={() => setExpanded(prev => !prev)}
      >
        <span>🎯 Reflection Reranking Log</span>
        <span className="rerank-log-summary">
          {total} evaluated → {kept.length} selected
        </span>
        <span style={{ marginLeft: 'auto' }}>{expanded ? '▾' : '▸'}</span>
      </button>

      {expanded && (
        <div className="rerank-log-body">
          {explanation && (
            <div className="rerank-explanation-box">
              ⚠️ {explanation}
            </div>
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
                  <tr key={`keep-${i}`} className={d.is_borderline ? 'row-borderline' : 'row-keep'}>
                    <td>#{d.final_rank}</td>
                    <td className="cell-description">{(d.hadron_id || '—')}</td>
                    <td>{d.rerank_score?.toFixed(2)}</td>
                    <td>
                      {d.is_borderline
                        ? <span className="decision-borderline">⚠ Borderline</span>
                        : <span className="decision-keep">✓ Keep</span>
                      }
                    </td>
                    <td className="cell-reason">{d.reason || '—'}</td>
                    <td>{d.confidence != null ? `${(d.confidence * 100).toFixed(0)}%` : '—'}</td>
                  </tr>
                ))
              }
              {/* Discarded results */}
              {discarded.map((d, i) => (
                <tr key={`discard-${i}`} className="row-discard">
                  <td>—</td>
                  <td className="cell-description">{d.hadron_id || '—'}</td>
                  <td>{d.rerank_score?.toFixed(2)}</td>
                  <td><span className="decision-discard">✗ Discard</span></td>
                  <td className="cell-reason">{d.reason || '—'}</td>
                  <td>{d.confidence != null ? `${(d.confidence * 100).toFixed(0)}%` : '—'}</td>
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
    setExpandedSteps(prev => ({ ...prev, [idx]: !prev[idx] }));
  };

  const agentIcons = {
    'Squad Router': '🧭',
    'Project Manager': '📋',
    'Search Specialist': '🔍',
    'Synthesizer': '✨',
    'Reflection Reranker': '🎯'
  };

  const agentColors = {
    'Squad Router': '#6366f1',
    'Project Manager': '#f59e0b',
    'Search Specialist': '#10b981',
    'Synthesizer': '#8b5cf6',
    'Reflection Reranker': '#0d9488'
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
        <span className="workflow-toggle-icon">{expanded ? '▾' : '▸'}</span>
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
                  style={{ borderColor: agentColors[step.agent] || '#6b7280' }}
                >
                  <span>{agentIcons[step.agent] || '⚙️'}</span>
                  <span>{step.agent}</span>
                </div>
                {idx < steps.length - 1 && <div className="workflow-flow-arrow">→</div>}
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
                      style={{ background: agentColors[step.agent] || '#6b7280' }}
                    />
                    <span className="workflow-step-agent">{agentIcons[step.agent] || '⚙️'} {step.agent}</span>
                    <span className="workflow-step-action">— {step.action}</span>
                  </div>
                  <span className="workflow-step-expand">
                    {expandedSteps[idx] ? '▾' : '▸'}
                  </span>
                </div>

                <div className="workflow-step-reasoning">
                  💭 {step.reasoning}
                  {step.opensearch_payload && (
                    <button
                      className="payload-link"
                      onClick={(e) => { e.stopPropagation(); setPayloadModal({ title: `${step.agent} — OpenSearch Payload`, payload: step.opensearch_payload, url: step.opensearch_url }); }}
                    >
                      📋 View OpenSearch Payload
                    </button>
                  )}
                  {step.search_service_response && (
                    <button
                      className="payload-link"
                      onClick={(e) => { e.stopPropagation(); setPayloadModal({ title: `Search Service Response — ${step.action}`, payload: step.search_service_response, url: step.search_service_endpoint, method: 'GET' }); }}
                    >
                      🌐 View Search Service Response
                    </button>
                  )}
                </div>

                {expandedSteps[idx] && (
                  <div className="workflow-step-details">
                    {step.decision && (
                      <div className="workflow-detail-block">
                        <div className="workflow-detail-label">🔀 Route Decision</div>
                        <code>{step.decision}</code>
                      </div>
                    )}
                    {step.prompt && (
                      <div className="workflow-detail-block">
                        <div className="workflow-detail-label">📝 System Prompt</div>
                        <pre className="workflow-prompt">{step.prompt}</pre>
                      </div>
                    )}
                    {step.input && (
                      <div className="workflow-detail-block">
                        <div className="workflow-detail-label">📥 Input</div>
                        <pre className="workflow-json">{JSON.stringify(step.input, null, 2)}</pre>
                      </div>
                    )}
                    {step.output && (
                      <div className="workflow-detail-block">
                        <div className="workflow-detail-label">📤 Output</div>
                        <pre className="workflow-json">{JSON.stringify(step.output, null, 2)}</pre>
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
  const [inputMessage, setInputMessage] = useState('');
  const [conversationId, setConversationId] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  // True when the outgoing message contains a rerank trigger phrase
  const [isReranking, setIsReranking] = useState(false);
  const [showApiKeyModal, setShowApiKeyModal] = useState(false);
  const [apiKey, setApiKey] = useState('');
  const [tempApiKey, setTempApiKey] = useState('');
  const [error, setError] = useState(null);

  // Check for API key on mount
  useEffect(() => {
    const initializeApp = async () => {
      const storedApiKey = sessionStorage.getItem('openai_api_key');
      if (!storedApiKey) {
        setShowApiKeyModal(true);
      } else {
        setApiKey(storedApiKey);
      }

      await loadRecentConversations();

      // Restore the last active conversation after page reload.
      const storedConversationId = sessionStorage.getItem(ACTIVE_CONVERSATION_KEY);
      if (storedConversationId) {
        const loaded = await loadConversation(storedConversationId);
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
      console.error('Failed to load conversations:', err);
      return [];
    }
  };

  const loadConversation = async (convId) => {
    try {
      setIsLoading(true);
      const conversation = await conversationsAPI.getConversation(convId);
      
      // Transform backend messages to frontend format
      const loadedMessages = [];
      if (conversation.messages && conversation.messages.length > 0) {
        conversation.messages.forEach(msg => {
          // Add user message
          loadedMessages.push({
            role: 'user',
            content: msg.user_message,
            file: conversation.file_name || null
          });
          // Add assistant message
          loadedMessages.push({
            role: 'assistant',
            content: msg.agent_response,
            results: msg.search_results_count || null
          });
        });
      }
      
      setMessages(loadedMessages);
      setConversationId(convId);
      sessionStorage.setItem(ACTIVE_CONVERSATION_KEY, convId);
      setSelectedFile(null);
      return true;
    } catch (err) {
      console.error('Failed to load conversation:', err);
      showError('Failed to load conversation history');
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  const handleApiKeySubmit = () => {
    if (tempApiKey.trim()) {
      sessionStorage.setItem('openai_api_key', tempApiKey);
      setApiKey(tempApiKey);
      setShowApiKeyModal(false);
      setTempApiKey('');
    }
  };

  // Focus input when modal opens
  useEffect(() => {
    if (showApiKeyModal) {
      setTimeout(() => {
        const input = document.querySelector('.modal-input');
        if (input) input.focus();
      }, 100);
    }
  }, [showApiKeyModal]);

  const handleNewChat = () => {
    setConversationId(null);
    sessionStorage.removeItem(ACTIVE_CONVERSATION_KEY);
    setMessages([]);
    setSelectedFile(null);
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
      console.error('Failed to delete conversation:', err);
      showError('Failed to delete conversation');
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (file.size > 1024 * 1024) {
        showError('File size exceeds 1MB limit');
        return;
      }
      setSelectedFile(file);
    }
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() && !selectedFile) return;
    if (!apiKey && !conversationId) {
      setShowApiKeyModal(true);
      return;
    }

    const userMessage = inputMessage.trim();
    setInputMessage('');

    // Detect rerank trigger phrases to show the dedicated loading indicator
    const rerankTrigger = /\bbest\b|\btop[\s-]?ranked?\b|\brerank\b|\breflect\s+and\s+respond\b|\breviewed\b/i;
    setIsReranking(rerankTrigger.test(userMessage));

    // Add user message to UI
    const newUserMessage = { 
      role: 'user', 
      content: userMessage,
      file: selectedFile?.name 
    };
    setMessages(prev => [...prev, newUserMessage]);

    setIsLoading(true);

    try {
      const response = await chatAPI.sendMessage(
        userMessage,
        conversationId,
        apiKey,  // Always send API key to extend session
        selectedFile
      );

      // Add agent response
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: response.response,
        results: response.results,
        filter_metadata: response.filter_metadata || null,
        workflow_steps: response.workflow_steps || [],
        search_mode: response.search_mode || 'relevance',
        rerank_applied: response.rerank_applied || false,
        rerank_decisions: response.rerank_decisions || [],
        rerank_explanation: response.rerank_explanation || null,
      }]);

      // Update conversation ID if new
      if (!conversationId) {
        setConversationId(response.conversation_id);
        sessionStorage.setItem(ACTIVE_CONVERSATION_KEY, response.conversation_id);
      }

      // Clear file selection
      setSelectedFile(null);

      // Reload conversations list
      await loadRecentConversations();

      // Handle expired API key
      if (!response.api_key_valid) {
        sessionStorage.removeItem('openai_api_key');
        setApiKey('');
        setShowApiKeyModal(true);
        showError('Session expired. Please enter your API key again.');
      }

    } catch (err) {
      console.error('Chat error:', err);
      console.error('Error status:', err.response?.status);
      console.error('Error detail:', err.response?.data?.detail);
      
      // Remove the user message that was just added since it failed
      setMessages(prev => prev.slice(0, -1));
      
      if (err.response?.status === 401) {
        console.log('Authentication error detected - clearing key and showing modal');
        // Clear invalid API key
        sessionStorage.removeItem('openai_api_key');
        setApiKey('');
        setTempApiKey('');
        
        const errorMsg = err.response?.data?.detail || 'Invalid API key. Please enter a valid OpenAI API key.';
        showError(errorMsg);
        
        // Force modal to show after state updates
        setTimeout(() => {
          setShowApiKeyModal(true);
          console.log('Modal should be visible now');
        }, 0);
      } else {
        showError(err.response?.data?.detail || 'Failed to send message');
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
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="app">
      {/* Sidebar */}
      <div className="sidebar">
        <h2>Recent Conversations</h2>
        <div className="conversation-list">
          {conversations.map(conv => (
            <div
              key={conv.conversation_id}
              className={`conversation-item ${conv.conversation_id === conversationId ? 'active' : ''}`}
              onClick={() => loadConversation(conv.conversation_id)}
            >
              <div className="conversation-item-content">
                <div className="conversation-query">{conv.title || conv.last_user_query || 'New conversation'}</div>
                <div className="conversation-meta">
                  {conv.message_count} message{conv.message_count !== 1 ? 's' : ''}
                </div>
              </div>
              <button
                className="conversation-delete-btn"
                title="Delete conversation"
                onClick={(e) => handleDeleteConversation(e, conv.conversation_id)}
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
          <button className="new-chat-btn" onClick={handleNewChat}>
            + New Chat
          </button>
        </div>

        <div className="messages-area">
          {messages.length === 0 && (
            <div style={{ textAlign: 'center', color: '#999', marginTop: '4rem' }}>
              <h2>Welcome to Gen-Aperture</h2>
              <p>Start a conversation to search for stock photos</p>
            </div>
          )}
          
          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              <div className="message-avatar">
                {msg.role === 'user' ? 'U' : 'A'}
              </div>
              <div className="message-content">
                {msg.role === 'assistant' ? (
                  <div className="markdown-body">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                  </div>
                ) : (
                  <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                )}
                {msg.file && (
                  <div style={{ marginTop: '0.5rem', fontSize: '0.875rem', color: '#666' }}>
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
                
                {/* Image results - show up to 10 with filters applied inline */}
                {msg.results && msg.results.length > 0 && (() => {
                  const activeResults = msg.results;
                  return (
                    <div className="image-results">
                      <div className="image-results-header">
                        📸 Showing {Math.min(activeResults.length, 10)} images
                        {msg.search_mode && (
                          <span className={`search-mode-badge ${msg.search_mode}`}>
                            {msg.search_mode === 'popular' ? '🔥 Popular' : '🎯 Relevant'}
                          </span>
                        )}
                        {msg.rerank_applied && (
                          <span className="rerank-badge">🎯 Reranked</span>
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
                        {activeResults.slice(0, 10).map((result, resultIdx) => (
                          <div key={resultIdx} className={`image-card${result.is_generated ? ' image-card--ai-generated' : ''}`}>
                            <img 
                              src={result.thumbnail_url} 
                              alt={result.description}
                              loading="lazy"
                              onClick={() => window.open(result.image_url, '_blank')}
                              style={{ cursor: 'pointer' }}
                            />
                            {result.is_generated && (
                              <div className="ai-generated-badge">✨ AI Generated</div>
                            )}
                            <div className="image-info">
                              <div className="image-description" title={result.description}>
                                {result.description}
                              </div>
                              <div className="image-meta">
                                <span>🏆 {result.license_count || 0} licenses</span>
                                {result.score && (
                                  <span>⭐ {result.score.toFixed(2)}</span>
                                )}
                              </div>
                            </div>
                          </div>
                        ))}
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
                  <p className="rerank-loading-text">🔄 Applying reflection reranking…</p>
                )}
              </div>
            </div>
          )}
        </div>

        <div className="input-area">
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
                {isLoading ? <div className="loading"></div> : '→'}
              </button>
            </div>
          </div>

          {selectedFile && (
            <div className="file-preview">
              <div className="file-preview-info">
                📎 {selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)
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

      {/* API Key Modal */}
      {showApiKeyModal && (
        <div className="modal-overlay">
          <div className="modal">
            <h2>OpenAI API Key Required</h2>
            <p>
              Please enter your OpenAI API key to use Gen-Aperture. Your key will be stored
              in your browser session (30 minutes) and never saved on our servers.
            </p>
            <input
              type="password"
              className="modal-input"
              placeholder="sk-..."
              value={tempApiKey}
              onChange={(e) => setTempApiKey(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleApiKeySubmit()}
            />
            <div className="modal-actions">
              <button
                className="modal-btn primary"
                onClick={handleApiKeySubmit}
                disabled={!tempApiKey.trim()}
              >
                Continue
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Error Toast */}
      {error && (
        <div className="toast">
          {error}
        </div>
      )}
    </div>
  );
}

export default App;
