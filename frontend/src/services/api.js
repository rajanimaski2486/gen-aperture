import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

export const api = axios.create({
  baseURL: API_BASE_URL,
});

export const chatAPI = {
  sendMessage: async (message, conversationId, file, model, workflowMode = 'agent_squad') => {
    const formData = new FormData();
    formData.append('message', message);
    formData.append('workflow_mode', workflowMode);
    
    if (conversationId) {
      formData.append('conversation_id', conversationId);
    }
    
    if (file) {
      formData.append('file', file);
    }

    if (model) {
      formData.append('model', model);
    }
    
    const response = await api.post('/chat', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  },
};

export const conversationsAPI = {
  getRecent: async () => {
    const response = await api.get('/conversations/recent');
    return response.data;
  },
  
  getConversation: async (conversationId) => {
    const response = await api.get(`/conversations/${conversationId}`);
    return response.data;
  },

  deleteConversation: async (conversationId) => {
    const response = await api.delete(`/conversations/${conversationId}`);
    return response.data;
  },
};
