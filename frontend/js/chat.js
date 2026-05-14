/**
 * Trợ Lý Luật Giao Thông - Chat Application
 * ChatGPT-inspired interface for traffic law Q&A
 */

(function () {
  'use strict';

  // =========================================================================
  // Configuration
  // =========================================================================

  const API_BASE = window.location.origin;
  const ENDPOINTS = {
    ask: API_BASE + '/ask',
    signImage: API_BASE + '/signs/image',
  };

  // =========================================================================
  // ChatApp — main application controller
  // =========================================================================

  class ChatApp {
    constructor() {
      // State
      this.conversations = [];
      this.currentConversationId = null;
      this.isLoading = false;
      this.abortController = null;

      // DOM elements
      this.elements = {
        chatMessages: document.getElementById('chatMessages'),
        chatInput: document.getElementById('chatInput'),
        sendBtn: document.getElementById('sendBtn'),
        newChatBtn: document.getElementById('newChatBtn'),
        welcomeScreen: document.getElementById('welcomeScreen'),
        conversationList: document.getElementById('conversationList'),
        sidebar: document.getElementById('sidebar'),
        toggleSidebarBtn: document.getElementById('toggleSidebarBtn'),
        mobileMenuBtn: document.getElementById('mobileMenuBtn'),
        chatTitle: document.getElementById('chatTitle'),
        suggestionGrid: document.getElementById('suggestionGrid'),
      };

      // Templates
      this.templates = {
        messageAi: document.getElementById('messageAiTemplate'),
        messageUser: document.getElementById('messageUserTemplate'),
        signCard: document.getElementById('signCardTemplate'),
        citation: document.getElementById('citationTemplate'),
      };

      // Bind methods
      this._onSend = this._onSend.bind(this);
      this._onInputKeydown = this._onInputKeydown.bind(this);
      this._onInputInput = this._onInputInput.bind(this);
      this._onNewChat = this._onNewChat.bind(this);
      this._onToggleSidebar = this._onToggleSidebar.bind(this);
      this._onMobileMenu = this._onMobileMenu.bind(this);
      this._onSuggestionClick = this._onSuggestionClick.bind(this);

      // Initialize
      this._init();
    }

    // =======================================================================
    // Initialization
    // =======================================================================

    _init() {
      this._bindEvents();
      this._loadConversations();
      this._updateSendButton();
      this._focusInput();

      // If no conversations exist, show welcome screen
      if (this.conversations.length === 0) {
        this._createConversation();
      } else {
        this._switchConversation(this.conversations[0].id);
      }
    }

    _bindEvents() {
      const { sendBtn, chatInput, newChatBtn, toggleSidebarBtn, mobileMenuBtn, suggestionGrid } = this.elements;

      sendBtn.addEventListener('click', this._onSend);
      chatInput.addEventListener('keydown', this._onInputKeydown);
      chatInput.addEventListener('input', this._onInputInput);
      newChatBtn.addEventListener('click', this._onNewChat);
      toggleSidebarBtn.addEventListener('click', this._onToggleSidebar);
      mobileMenuBtn.addEventListener('click', this._onMobileMenu);
      suggestionGrid.addEventListener('click', this._onSuggestionClick);
    }

    _focusInput() {
      // Small delay to ensure DOM is ready
      setTimeout(() => this.elements.chatInput.focus(), 100);
    }

    // =======================================================================
    // Conversation management
    // =======================================================================

    _createConversation(title) {
      const conv = {
        id: this._generateId(),
        title: title || 'Cuộc trò chuyện mới',
        messages: [],
        createdAt: Date.now(),
      };

      this.conversations.unshift(conv);
      this._saveConversations();
      this._switchConversation(conv.id);
      return conv;
    }

    _switchConversation(convId) {
      this.currentConversationId = convId;
      const conv = this._getCurrentConversation();

      // Update title
      this.elements.chatTitle.textContent = conv ? conv.title : 'Trợ Lý Luật Giao Thông';

      // Re-render all messages
      this._renderAllMessages();

      // Update conversation list sidebar
      this._renderConversationList();

      // Show/hide welcome screen
      if (conv && conv.messages.length === 0) {
        this.elements.welcomeScreen.classList.remove('hidden');
      } else {
        this.elements.welcomeScreen.classList.add('hidden');
      }

      this._scrollToBottom();
      this._focusInput();
    }

    _getCurrentConversation() {
      return this.conversations.find(c => c.id === this.currentConversationId) || null;
    }

    _deleteConversation(convId) {
      const idx = this.conversations.findIndex(c => c.id === convId);
      if (idx === -1) return;

      this.conversations.splice(idx, 1);
      this._saveConversations();

      if (this.currentConversationId === convId) {
        if (this.conversations.length > 0) {
          this._switchConversation(this.conversations[0].id);
        } else {
          this._createConversation();
        }
      } else {
        this._renderConversationList();
      }
    }

    _updateConversationTitle(convId, firstMessage) {
      const conv = this.conversations.find(c => c.id === convId);
      if (!conv || conv.title !== 'Cuộc trò chuyện mới') return;

      // Use first message as title, truncated
      const title = firstMessage.length > 40 ? firstMessage.substring(0, 40) + '...' : firstMessage;
      conv.title = title;
      this.elements.chatTitle.textContent = title;
      this._saveConversations();
      this._renderConversationList();
    }

    _generateId() {
      return 'conv_' + Date.now() + '_' + Math.random().toString(36).substring(2, 8);
    }

    _saveConversations() {
      try {
        const data = this.conversations.map(c => ({
          id: c.id,
          title: c.title,
          messages: c.messages,
          createdAt: c.createdAt,
        }));
        localStorage.setItem('traffic_law_conversations', JSON.stringify(data));
      } catch (e) {
        // localStorage may be full or unavailable
      }
    }

    _loadConversations() {
      try {
        const raw = localStorage.getItem('traffic_law_conversations');
        if (raw) {
          this.conversations = JSON.parse(raw);
        }
      } catch (e) {
        this.conversations = [];
      }
    }

    // =======================================================================
    // Sending messages
    // =======================================================================

    async _onSend() {
      if (this.isLoading) {
        // Abort current request
        if (this.abortController) {
          this.abortController.abort();
        }
        return;
      }

      const question = this.elements.chatInput.value.trim();
      if (!question) return;

      // Clear input
      this.elements.chatInput.value = '';
      this._autoResizeInput();
      this._updateSendButton();

      // Hide welcome screen
      this.elements.welcomeScreen.classList.add('hidden');

      // Get or create conversation
      let conv = this._getCurrentConversation();
      if (!conv) {
        conv = this._createConversation();
      }

      // Update title on first message
      if (conv.messages.length === 0) {
        this._updateConversationTitle(conv.id, question);
      }

      // Add user message
      this._addMessage('user', question);
      this._renderAllMessages();
      this._scrollToBottom();
      this._renderConversationList();

      // Show loading indicator
      this._setLoading(true);
      this._renderAllMessages();
      this._scrollToBottom();

      // Call API
      try {
        this.abortController = new AbortController();

        const response = await fetch(ENDPOINTS.ask, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question, k: 5 }),
          signal: this.abortController.signal,
        });

        if (!response.ok) {
          throw new Error(`Server error: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();

        if (!data.ok) {
          throw new Error(data.error || 'Unknown error');
        }

        // Add AI message with full response data
        this._addMessage('ai', data.answer, {
          signs: data.signs || [],
          citations: data.citations || [],
          extracted: data.extracted || {},
        });

      } catch (err) {
        if (err.name === 'AbortError') {
          this._addMessage('ai', '⏹ _Đã dừng trả lời._');
        } else {
          this._addMessage('ai', null, { error: err.message });
        }
      } finally {
        this.abortController = null;
        this._setLoading(false);
        this._renderAllMessages();
        this._scrollToBottom();
        this._focusInput();
      }
    }

    _addMessage(role, content, meta = {}) {
      const conv = this._getCurrentConversation();
      if (!conv) return;

      conv.messages.push({
        id: this._generateId(),
        role,
        content,
        signs: meta.signs || [],
        citations: meta.citations || [],
        extracted: meta.extracted || {},
        error: meta.error || null,
        timestamp: Date.now(),
      });
    }

    _setLoading(isLoading) {
      this.isLoading = isLoading;
      const { sendBtn } = this.elements;

      if (isLoading) {
        sendBtn.classList.add('sending');
        sendBtn.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
            <rect x="4" y="4" width="16" height="16" rx="2" />
          </svg>`;
      } else {
        sendBtn.classList.remove('sending');
        sendBtn.innerHTML = `
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="22" y1="2" x2="11" y2="13"></line>
            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
          </svg>`;
      }
    }

    // =======================================================================
    // Rendering
    // =======================================================================

    _renderAllMessages() {
      const { chatMessages } = this.elements;
      const conv = this._getCurrentConversation();

      // Clear existing messages (keep welcome screen)
      chatMessages.querySelectorAll('.message, .typing-message').forEach(el => el.remove());

      if (!conv || conv.messages.length === 0) {
        this.elements.welcomeScreen.classList.remove('hidden');
        return;
      }

      this.elements.welcomeScreen.classList.add('hidden');

      // Render each message
      conv.messages.forEach(msg => {
        const el = this._renderMessage(msg);
        chatMessages.appendChild(el);
      });

      // Render loading indicator if needed
      if (this.isLoading) {
        const loadingEl = this._createLoadingIndicator();
        chatMessages.appendChild(loadingEl);
      }
    }

    _renderMessage(msg) {
      if (msg.role === 'user') {
        return this._renderUserMessage(msg);
      }
      return this._renderAiMessage(msg);
    }

    _renderUserMessage(msg) {
      const clone = this.templates.messageUser.content.cloneNode(true);
      clone.querySelector('.message-text').textContent = msg.content;
      return clone.children[0];
    }

    _renderAiMessage(msg) {
      const clone = this.templates.messageAi.content.cloneNode(true);
      const contentEl = clone.querySelector('.message-content');
      const textEl = contentEl.querySelector('.message-text');
      const signsEl = contentEl.querySelector('.message-signs');
      const citationsEl = contentEl.querySelector('.message-citations');
      const copyBtn = contentEl.querySelector('.copy-btn');

      if (msg.error) {
        // Render error state
        textEl.innerHTML = this._renderError(msg.error);
      } else if (msg.content) {
        // Render markdown content
        textEl.innerHTML = this._renderMarkdown(msg.content);
      } else {
        textEl.innerHTML = '';
      }

      // Render extracted info (fines, points, suspension)
      const extractedEl = contentEl.querySelector('.message-extracted');
      if (msg.extracted && this._hasExtractedInfo(msg.extracted)) {
        extractedEl.innerHTML = this._renderExtracted(msg.extracted);
        extractedEl.style.display = '';
      } else {
        extractedEl.style.display = 'none';
      }

      // Render sign cards
      if (msg.signs && msg.signs.length > 0) {
        signsEl.innerHTML = '';
        msg.signs.forEach(sign => {
          signsEl.appendChild(this._renderSignCard(sign));
        });
      } else {
        signsEl.style.display = 'none';
      }

      // Render citations
      if (msg.citations && msg.citations.length > 0) {
        citationsEl.innerHTML = '<div class="citations-header">📋 Căn cứ pháp lý</div>';
        msg.citations.forEach(cit => {
          citationsEl.appendChild(this._renderCitation(cit));
        });
      } else {
        citationsEl.style.display = 'none';
      }

      // Copy button
      copyBtn.addEventListener('click', () => this._copyMessage(msg, copyBtn));

      return clone.children[0];
    }

    _renderMarkdown(text) {
      if (typeof marked !== 'undefined' && marked.parse) {
        try {
          const html = marked.parse(text);
          return html;
        } catch (e) {
          // Fallback to simple formatting
        }
      }
      // Fallback: escape HTML and convert newlines to <br>
      return this._escapeHtml(text).replace(/\n/g, '<br>');
    }

    _renderError(errorMsg) {
      return `
        <div class="error-banner">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
          <span>${this._escapeHtml(errorMsg)}</span>
          <button class="retry-btn" onclick="window.chatApp._retryLast()">Thử lại</button>
        </div>`;
    }

    _renderSignCard(sign) {
      const clone = this.templates.signCard.content.cloneNode(true);
      const img = clone.querySelector('.sign-image');
      const fallback = clone.querySelector('.sign-image-fallback');
      const nameEl = clone.querySelector('.sign-name');
      const categoryEl = clone.querySelector('.sign-category');
      const descEl = clone.querySelector('.sign-description');

      nameEl.textContent = sign.name || '';
      const scorePct = sign.score ? ' · Độ khớp ' + Math.round(sign.score * 100) + '%' : '';
      categoryEl.textContent = (sign.category_name || '') + scorePct;
      descEl.textContent = sign.description || '';

      // Load image
      if (sign.image_exists && sign.name) {
        const imgUrl = ENDPOINTS.signImage + '?name=' + encodeURIComponent(sign.name);
        img.src = imgUrl;
        img.onerror = () => {
          img.style.display = 'none';
          fallback.style.display = 'flex';
        };
        img.onload = () => {
          img.style.display = 'block';
          fallback.style.display = 'none';
        };
      } else {
        img.style.display = 'none';
        fallback.style.display = 'flex';
      }

      return clone.children[0];
    }

    _hasExtractedInfo(extracted) {
      const hasFines = extracted.fines && extracted.fines.length > 0
        && !extracted.fines[0].includes('Không tìm thấy');
      const hasPoints = extracted.points && extracted.points.length > 0
        && !extracted.points[0].includes('Không tìm thấy');
      const hasSuspension = extracted.suspension && extracted.suspension.length > 0;
      return hasFines || hasPoints || hasSuspension;
    }

    _renderExtracted(extracted) {
      let html = '';

      if (extracted.fines && extracted.fines.length > 0
        && !extracted.fines[0].includes('Không tìm thấy')) {
        html += '<div class="extracted-badge fine">'
          + '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>'
          + '<span class="badge-label">Mức phạt</span>'
          + '<span class="badge-value">' + this._escapeHtml(extracted.fines[0]) + '</span>'
          + '</div>';
      }

      if (extracted.points && extracted.points.length > 0
        && !extracted.points[0].includes('Không tìm thấy')) {
        html += '<div class="extracted-badge points">'
          + '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>'
          + '<span class="badge-label">Trừ điểm</span>'
          + '<span class="badge-value">' + this._escapeHtml(extracted.points[0]) + '</span>'
          + '</div>';
      }

      if (extracted.suspension && extracted.suspension.length > 0) {
        html += '<div class="extracted-badge suspension">'
          + '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>'
          + '<span class="badge-label">Tước bằng</span>'
          + '<span class="badge-value">' + this._escapeHtml(extracted.suspension[0]) + '</span>'
          + '</div>';
      }

      return html;
    }

    _renderCitation(cit) {
      const clone = this.templates.citation.content.cloneNode(true);
      clone.querySelector('.citation-source').textContent =
        (cit.source || 'Nguồn') + (cit.page ? ' - Trang ' + cit.page : '');
      clone.querySelector('.citation-score').textContent =
        cit.score ? 'Độ khớp: ' + (cit.score * 100).toFixed(0) + '%' : '';
      clone.querySelector('.citation-text').textContent =
        (cit.text || '').substring(0, 300);
      return clone.children[0];
    }

    _createLoadingIndicator() {
      const el = document.createElement('div');
      el.className = 'message message-ai typing-message';
      el.innerHTML = `
        <div class="message-avatar">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 22s-8-4.5-8-11.8V5.2L12 3l8 2.2v5c0 7.3-8 11.8-8 11.8z"/>
            <path d="M9 12l2 2 4-4"/>
          </svg>
        </div>
        <div class="message-content">
          <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
          </div>
        </div>`;
      return el;
    }

    _renderConversationList() {
      const { conversationList } = this.elements;

      conversationList.innerHTML = this.conversations.map(conv => {
        const isActive = conv.id === this.currentConversationId;
        return `
          <div class="conversation-item${isActive ? ' active' : ''}"
               data-conv-id="${conv.id}"
               onclick="window.chatApp._switchConversation('${conv.id}')">
            <svg class="conv-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
            </svg>
            <span class="conv-title">${this._escapeHtml(conv.title)}</span>
            <button class="conv-delete"
                    title="Xóa"
                    onclick="event.stopPropagation(); window.chatApp._deleteConversation('${conv.id}')">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><line x1="10" y1="11" x2="10" y2="17"/><line x1="14" y1="11" x2="14" y2="17"/>
              </svg>
            </button>
          </div>`;
      }).join('');
    }

    // =======================================================================
    // Actions
    // =======================================================================

    _copyMessage(msg, btn) {
      const text = msg.content || '';
      navigator.clipboard.writeText(text).then(() => {
        btn.classList.add('copied');
        setTimeout(() => btn.classList.remove('copied'), 2000);
      }).catch(() => {
        // Fallback for clipboard API not available
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        btn.classList.add('copied');
        setTimeout(() => btn.classList.remove('copied'), 2000);
      });
    }

    _retryLast() {
      const conv = this._getCurrentConversation();
      if (!conv || conv.messages.length === 0) return;

      // Find last user message
      const lastUserMsg = [...conv.messages].reverse().find(m => m.role === 'user');
      if (!lastUserMsg) return;

      // Remove last AI message (the error one)
      const lastIdx = conv.messages.length - 1;
      if (conv.messages[lastIdx].role === 'ai') {
        conv.messages.pop();
      }

      // Re-send
      this.elements.chatInput.value = lastUserMsg.content;
      this._onSend();
    }

    _onNewChat() {
      this._createConversation();
      this._focusInput();

      // Close mobile sidebar if open
      this._closeMobileSidebar();
    }

    _onToggleSidebar() {
      this.elements.sidebar.classList.toggle('collapsed');
    }

    _onMobileMenu() {
      const { sidebar } = this.elements;

      // Create overlay if doesn't exist
      let overlay = document.querySelector('.sidebar-overlay');
      if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'sidebar-overlay';
        overlay.addEventListener('click', () => this._closeMobileSidebar());
        document.body.appendChild(overlay);
      }

      sidebar.classList.toggle('mobile-open');
      overlay.classList.toggle('visible');
    }

    _closeMobileSidebar() {
      this.elements.sidebar.classList.remove('mobile-open');
      const overlay = document.querySelector('.sidebar-overlay');
      if (overlay) {
        overlay.classList.remove('visible');
      }
    }

    _onSuggestionClick(e) {
      const card = e.target.closest('.suggestion-card');
      if (!card) return;

      const query = card.dataset.query;
      if (query) {
        this.elements.chatInput.value = query;
        this._autoResizeInput();
        this._updateSendButton();
        this._onSend();
      }
    }

    // =======================================================================
    // Input handling
    // =======================================================================

    _onInputKeydown(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this._onSend();
      }
    }

    _onInputInput() {
      this._autoResizeInput();
      this._updateSendButton();
    }

    _autoResizeInput() {
      const input = this.elements.chatInput;
      input.style.height = 'auto';
      input.style.height = Math.min(input.scrollHeight, 200) + 'px';
    }

    _updateSendButton() {
      const hasText = this.elements.chatInput.value.trim().length > 0;
      this.elements.sendBtn.disabled = !hasText && !this.isLoading;
    }

    // =======================================================================
    // Utilities
    // =======================================================================

    _scrollToBottom() {
      requestAnimationFrame(() => {
        this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
      });
    }

    _escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }
  }

  // =========================================================================
  // Boot
  // =========================================================================

  document.addEventListener('DOMContentLoaded', () => {
    window.chatApp = new ChatApp();
  });

})();
