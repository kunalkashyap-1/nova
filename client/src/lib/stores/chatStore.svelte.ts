import { v4 as uuidv4 } from 'uuid';

// SSE Event Interfaces
export interface SSEChatBaseEvent {
  type: 'init' | 'message' | 'title_update';
  id: string;
}

export interface SSEChatInitEvent extends SSEChatBaseEvent {
  type: 'init';
  title: string;
}

export interface SSEChatMessageEvent extends SSEChatBaseEvent {
  type: 'message';
  message: Message;
}

export interface SSEChatTitleUpdateEvent extends SSEChatBaseEvent {
  type: 'title_update';
  title: string;
}

export type SSEChatEvent = SSEChatInitEvent | SSEChatMessageEvent | SSEChatTitleUpdateEvent;

export type Message = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
};

export type Conversation = {
  id: string;
  title: string;
  updatedAt: string;
};

export type ChatState = {
  conversations: Conversation[];
  messages: Message[];
  activeConversationId: string | null;
  isLoading: boolean;
  errorMessage: string;
  user: {
    username: string;
    profilePicUrl: string;
  };
  selectedModel: string;
};

export const chatState: ChatState = $state({
  conversations: [],
  messages: [],
  activeConversationId: null,
  isLoading: false,
  errorMessage: '',
  user: {
    username: 'NovaUser',
    profilePicUrl: ''
  },
  selectedModel: 'llama'
});

// --- User Auth State ---
export type AuthUser = {
  username: string;
  profilePicUrl?: string;
  isGuest: boolean;
};

function loadUserFromLocalStorage(): AuthUser {
  try {
    const raw = localStorage.getItem('nova_user');
    if (raw) {
      const parsed = JSON.parse(raw);
      return {
        username: parsed.username || 'Guest User',
        profilePicUrl: parsed.profilePicUrl || '',
        isGuest: !parsed.username,
      };
    }
  } catch {}
  return { username: 'Guest User', profilePicUrl: '', isGuest: true };
}

export const authUser: AuthUser = $state(loadUserFromLocalStorage());

export function setUser(user: Partial<AuthUser>) {
  const merged = { ...authUser, ...user, isGuest: !user.username };
  authUser.username = merged.username;
  authUser.profilePicUrl = merged.profilePicUrl || '';
  authUser.isGuest = merged.isGuest;
  localStorage.setItem('nova_user', JSON.stringify(merged));
}

export function signOut() {
  authUser.username = 'Guest User';
  authUser.profilePicUrl = '';
  authUser.isGuest = true;
  localStorage.removeItem('nova_user');
}

// --- SSE Mock Implementation ---
let mockConversations: Conversation[] = [];
let mockMessages: Record<string, Message[]> = {};

function createMockSSE(chatId: string, onEvent: (event: SSEChatEvent) => void, isNew: boolean, prompt?: string) {
  // Simulate streaming events
  if (isNew) {
    setTimeout(() => {
      onEvent({ type: 'init', id: chatId, title: prompt ? `Chat: ${prompt}` : 'New Chat' });
    }, 200);
    setTimeout(() => {
      const msg: Message = {
        id: uuidv4(),
        role: 'assistant',
        content: `Hello! This is a new chat for: "${prompt}"`,
        timestamp: new Date().toISOString()
      };
      onEvent({ type: 'message', id: chatId, message: msg });
      mockMessages[chatId] = [...(mockMessages[chatId] || []), msg];
    }, 600);
  } else {
    // Existing chat: stream all messages
    (mockMessages[chatId] || []).forEach((msg, idx) => {
      setTimeout(() => {
        onEvent({ type: 'message', id: chatId, message: msg });
      }, 200 * (idx + 1));
    });
  }
}

// --- SSE Client Logic (with fallback) ---
function connectToChatSSE(chatId: string, onEvent: (event: SSEChatEvent) => void, isNew = false, prompt?: string) {
  // For now, always use mock
  createMockSSE(chatId, onEvent, isNew, prompt);
}

function reconnectToChatSSE(chatId: string, onEvent: (event: SSEChatEvent) => void) {
  // For now, just re-stream previous messages
  createMockSSE(chatId, onEvent, false);
}

export function setLoading(isLoading: boolean) {
  chatState.isLoading = isLoading;
}

export function setError(message: string) {
  chatState.errorMessage = message;
  setTimeout(() => {
    if (chatState.errorMessage === message) {
      chatState.errorMessage = '';
    }
  }, 5000);
}

export async function loadConversations() {
  setLoading(true);
  try {
    // Use mock conversations for now
    chatState.conversations = mockConversations;
  } catch (e) {
    console.error('Error loading conversations:', e);
    setError('Failed to load conversations');
  } finally {
    setLoading(false);
  }
}

export async function loadConversation(id: string) {
  setLoading(true);
  chatState.messages = [];
  chatState.activeConversationId = id;
  try {
    connectToChatSSE(id, (event) => {
      if (event.type === 'init') {
        // Update conversation title and chat list
        let convo = chatState.conversations.find((c) => c.id === id);
        if (!convo) {
          convo = { id, title: event.title, updatedAt: new Date().toISOString() };
          chatState.conversations = [convo, ...chatState.conversations];
        } else {
          convo.title = event.title;
        }
      } else if (event.type === 'message') {
        chatState.messages = [...chatState.messages, event.message];
      } else if (event.type === 'title_update') {
        let convo = chatState.conversations.find((c) => c.id === id);
        if (convo) convo.title = event.title;
      }
    }, false);
  } catch (e) {
    console.error('Error loading conversation:', e);
    setError('Failed to load conversation messages');
  } finally {
    setLoading(false);
  }
}

export async function createNewChat(prompt: string): Promise<string | undefined> {
  if (!prompt.trim()) {
    setError('Please enter a chat prompt');
    return undefined;
  }
  setLoading(true);
  try {
    const newId = uuidv4();
    mockConversations = [
      { id: newId, title: `Chat: ${prompt}`, updatedAt: new Date().toISOString() },
      ...mockConversations
    ];
    // Add user message and initialize messages list
    const userMessage: Message = {
      id: uuidv4(),
      role: 'user',
      content: prompt,
      timestamp: new Date().toISOString()
    };
    mockMessages[newId] = [userMessage];
    chatState.activeConversationId = newId;
    chatState.messages = [userMessage];
    connectToChatSSE(newId, (event) => {
      if (event.type === 'init') {
        // Update conversation title and chat list
        let convo = chatState.conversations.find((c) => c.id === newId);
        if (!convo) {
          convo = { id: newId, title: event.title, updatedAt: new Date().toISOString() };
          chatState.conversations = [convo, ...chatState.conversations];
        } else {
          convo.title = event.title;
        }
      } else if (event.type === 'message') {
        chatState.messages = [...chatState.messages, event.message];
      }
    }, true, prompt);
    return newId;
  } catch (e) {
    console.error('Error creating new chat:', e);
    setError('Failed to create new chat');
    return undefined;
  } finally {
    setLoading(false);
  }
}

export async function sendMessage(content: string) {
  if (!content.trim() || !chatState.activeConversationId) return;
  const chatId = chatState.activeConversationId;
  const userMessage: Message = {
    id: uuidv4(),
    role: 'user',
    content,
    timestamp: new Date().toISOString()
  };
  chatState.messages.push(userMessage);
  chatState.isLoading = true;
  // Simulate streaming assistant reply
  setTimeout(() => {
    const assistantMessage: Message = {
      id: uuidv4(),
      role: 'assistant',
      content: `I received your message: "${content}". This is a simulated streaming response.`,
      timestamp: new Date().toISOString()
    };
    chatState.messages.push(assistantMessage);
    // Update chat list order
    const updated = chatState.conversations.filter((c) => c.id !== chatId);
    const active = chatState.conversations.find((c) => c.id === chatId);
    if (active) {
      active.updatedAt = new Date().toISOString();
      chatState.conversations = [active, ...updated];
    }
    // Simulate streaming more messages
    setTimeout(() => {
      const followup: Message = {
        id: uuidv4(),
        role: 'assistant',
        content: 'Here is a followup message streamed after a delay.',
        timestamp: new Date().toISOString()
      };
      chatState.messages.push(followup);
    }, 1200);
  }, 800);
  chatState.isLoading = false;
}

export function setSelectedModel(model: string) {
  chatState.selectedModel = model;
}

// Fallback HTTP polling (not implemented, stub)
export async function pollMessages(chatId: string) {
  // For fallback: fetch messages every N seconds
}
