import { useState, useEffect } from 'react';
import './index.css';
import { chatAPI, conversationsAPI } from './services/api';

/** Modal to display JSON payload */
function PayloadModal({ title, payload, onClose }) {
  if (!payload) return null;
  return (
    <div className="payload-modal-overlay" onClick={onClose}>
      <div className="payload-modal" onClick={e => e.stopPropagation()}>
        <div className="payload-modal-header">
          <h3>{title}</h3>
          <button className="payload-modal-close" onClick={onClose}>✕</button>
        </div>
        <pre className="payload-modal-body">{JSON.stringify(payload, null, 2)}</pre>
      </div>
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
    'Synthesizer': '✨'
  };

  const agentColors = {
    'Squad Router': '#6366f1',
    'Project Manager': '#f59e0b',
    'Search Specialist': '#10b981',
    'Synthesizer': '#8b5cf6'
  };

  return (
    <div className="workflow-panel">
      {payloadModal && (
        <PayloadModal
          title={payloadModal.title}
          payload={payloadModal.payload}
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
                      onClick={(e) => { e.stopPropagation(); setPayloadModal({ title: `${step.agent} — OpenSearch Payload`, payload: step.opensearch_payload }); }}
                    >
                      📋 View OpenSearch Payload
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
  const [showApiKeyModal, setShowApiKeyModal] = useState(false);
  const [apiKey, setApiKey] = useState('');
  const [tempApiKey, setTempApiKey] = useState('');
  const [error, setError] = useState(null);

  // Check for API key on mount
  useEffect(() => {
    const storedApiKey = sessionStorage.getItem('openai_api_key');
    if (!storedApiKey) {
      setShowApiKeyModal(true);
    } else {
      setApiKey(storedApiKey);
    }
    loadRecentConversations();
  }, []);

  const loadRecentConversations = async () => {
    try {
      const recent = await conversationsAPI.getRecent();
      setConversations(recent);
    } catch (err) {
      console.error('Failed to load conversations:', err);
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
      setSelectedFile(null);
    } catch (err) {
      console.error('Failed to load conversation:', err);
      showError('Failed to load conversation history');
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
    setMessages([]);
    setSelectedFile(null);
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
        workflow_steps: response.workflow_steps || [],
        search_mode: response.search_mode || 'relevance'
      }]);

      // Update conversation ID if new
      if (!conversationId) {
        setConversationId(response.conversation_id);
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
              <div className="conversation-query">{conv.last_user_query || 'New conversation'}</div>
              <div className="conversation-meta">
                {conv.message_count} message{conv.message_count !== 1 ? 's' : ''}
              </div>
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
                <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                {msg.file && (
                  <div style={{ marginTop: '0.5rem', fontSize: '0.875rem', color: '#666' }}>
                    📎 {msg.file}
                  </div>
                )}
                
                {/* Agent Workflow Panel */}
                {msg.workflow_steps && msg.workflow_steps.length > 0 && (
                  <AgentWorkflowPanel steps={msg.workflow_steps} />
                )}
                
                {/* Image results - show up to 10 */}
                {msg.results && msg.results.length > 0 && (
                  <div className="image-results">
                    <div className="image-results-header">
                      📸 Showing {Math.min(msg.results.length, 10)} images
                      {msg.search_mode && (
                        <span className={`search-mode-badge ${msg.search_mode}`}>
                          {msg.search_mode === 'popular' ? '🔥 Popular' : '🎯 Relevant'}
                        </span>
                      )}
                    </div>
                    <div className="image-grid">
                      {msg.results.slice(0, 10).map((result, resultIdx) => (
                        <div key={resultIdx} className="image-card">
                          <img 
                            src={result.thumbnail_url} 
                            alt={result.description}
                            loading="lazy"
                            onClick={() => window.open(result.image_url, '_blank')}
                            style={{ cursor: 'pointer' }}
                          />
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
                )}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="message assistant">
              <div className="message-avatar">A</div>
              <div className="message-content">
                <div className="loading"></div>
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
