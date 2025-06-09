import { v4 as uuidv4 } from 'uuid';
import api from '../api';
import { authStore } from './auth.svelte';

let currentAbortController: AbortController | null = null;

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
	meta?: { model?: string; time?: number };
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
	selectedModel: string;
	// Add stream state directly to the store
	isStreamActive: boolean;
};

export const chatState: ChatState = $state({
	conversations: [],
	messages: [],
	activeConversationId: null,
	isLoading: false,
	errorMessage: '',
	selectedModel: 'deepseek-r1:1.5b',
	isStreamActive: false
});

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

// --- Conversation Persistence ---
function saveConversationsToStorage(conversations: Conversation[]) {
	try {
		localStorage.setItem('nova_conversations', JSON.stringify(conversations));
	} catch {}
}

function loadConversationsFromStorage(): Conversation[] {
	try {
		const raw = localStorage.getItem('nova_conversations');
		if (raw) return JSON.parse(raw);
	} catch {}
	return [];
}

export async function loadConversations() {
	setLoading(true);
	try {
		chatState.conversations = loadConversationsFromStorage();
	} catch (e) {
		setError('Failed to load conversations');
	} finally {
		setLoading(false);
	}
}

export async function loadConversation(id: string) {
	setLoading(true);
	if (chatState.activeConversationId !== id) {
		chatState.messages = [];
	}
	chatState.activeConversationId = id;
	setLoading(false);
}

// Helper function to get user ID from auth store
function getUserId(): string | number {
	const user = authStore.user;
	// console.log(user);
	if (!user) {
		return 'guest';
	}
	return user.id || user.username || 'guest';
}

export async function createNewChat(prompt: string): Promise<string | undefined> {
	setLoading(true);
	try {
		// Get user ID from auth store
		const user_id = getUserId();
		
		// Create conversation on backend first
		try {
			const response = await api.post(`/conversations/`, {
				title: `Chat: ${prompt.slice(0, 30)}${prompt.length > 30 ? '...' : ''}`,
				user_id: user_id
			});

			// Get the created conversation from response
			const newConversation = response.data;
			const newId = newConversation.id;
			chatState.activeConversationId = newId;
			
			// Add conversation to list
			chatState.conversations = [newConversation, ...chatState.conversations];
			saveConversationsToStorage(chatState.conversations);
			
			// Create user message first
			const userMsg: Message = {
				id: uuidv4(),
				role: 'user',
				content: prompt,
				timestamp: new Date().toISOString()
			};
			chatState.messages = [userMsg];
			
			// Now connect to SSE after conversation is created
			connectToChatSSE(
				newId,
				prompt,
				(msg) => {},
				(meta) => {
					const conversation = chatState.conversations.find((c) => c.id === newId);
					if (conversation) {
						conversation.updatedAt = new Date().toISOString();
						chatState.conversations = [...chatState.conversations];
						saveConversationsToStorage(chatState.conversations);
					}
				}
			);
			return newId;
		} catch (error: any) {
			console.error('Error creating conversation:', error);
			if (error.response?.data?.detail?.[0]?.msg) {
				throw new Error(error.response.data.detail[0].msg);
			}
			throw new Error(error.message || 'Failed to create conversation on server');
		}
	} catch (e) {
		setError('Failed to create new chat');
		return undefined;
	} finally {
		setLoading(false);
	}
}

export async function sendMessage(content: string) {
	if (!content.trim() || !chatState.activeConversationId) return;
	const chatId = chatState.activeConversationId;
	const conversation = chatState.conversations.find((c) => c.id === chatId);
	if (conversation) {
		conversation.updatedAt = new Date().toISOString();
		chatState.conversations = [...chatState.conversations];
		saveConversationsToStorage(chatState.conversations);
	}
	connectToChatSSE(
		chatId,
		content,
		(msg) => {},
		(meta) => {
			const conversation = chatState.conversations.find((c) => c.id === chatId);
			if (conversation) {
				conversation.updatedAt = new Date().toISOString();
				chatState.conversations = [...chatState.conversations];
				saveConversationsToStorage(chatState.conversations);
			}
		}
	);
}

function connectToChatSSE(
    chatId: string,
    userMessage: string,
    onAssistantMessage: (msg: Message) => void,
    onFinal?: (meta: { model?: string; time?: number }) => void,
    onStopped?: () => void
) {
    // Create new abort controller for this request
    currentAbortController = new AbortController();
    
    // Update stream state immediately when starting
    chatState.isStreamActive = true;
    
    // Get user ID from auth store
    const user_id = getUserId();
    const model = chatState.selectedModel || 'deepseek-r1:1.5b';
    
    // Only add user message if it's not already the last message
    const lastMessage = chatState.messages[chatState.messages.length - 1];
    if (!lastMessage || lastMessage.content !== userMessage) {
        const userMsg: Message = {
            id: uuidv4(),
            role: 'user',
            content: userMessage,
            timestamp: new Date().toISOString()
        };
        chatState.messages = [...chatState.messages, userMsg];
    }
    
    let replyBuffer = '';
    let assistantMsg: Message | null = null;
    let meta: { model?: string; time?: number } = {};
    const preferences = JSON.parse(localStorage.getItem("nova_model_preferences") || '{}');
    
    fetch(`http://${import.meta.env.VITE_BACKEND_API_URL}/api/v1/messages/`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            chat_id: chatId,
            user_id,
            model,
            message: userMessage,
            provider: preferences.selectedProvider || 'ollama',
            stream: true,
            context_strategy: 'hybrid',
            optimize_context: true,
            max_context_docs: 15
        }),
        // Add abort signal
        signal: currentAbortController.signal
    }).then(async (response) => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
       
        const reader = response.body?.getReader();
        if (!reader) {
            throw new Error('No reader available');
        }
        const decoder = new TextDecoder();
        let done = false;
       
        try {
            while (!done) {
                const { value, done: doneReading } = await reader.read();
                done = doneReading;
                
                if (value) {
                    const chunk = decoder.decode(value, { stream: true });
                    chunk.split(/\n\n/).forEach((eventStr) => {
                        if (eventStr.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(eventStr.slice(6));
                               
                                // Handle errors
                                if (data.error) {
                                    console.error('Error from server:', data.error);
                                    chatState.errorMessage = data.error;
                                    return;
                                }
                                
                                const content = data.reply ?? '';
                                replyBuffer += content;
                                
                                if (!assistantMsg) {
                                    assistantMsg = {
                                        id: uuidv4(),
                                        role: 'assistant',
                                        content: '',
                                        timestamp: new Date().toISOString()
                                    };
                                    chatState.messages = [...chatState.messages, assistantMsg];
                                }
                                
                                assistantMsg = { ...assistantMsg, content: replyBuffer };
                                chatState.messages = chatState.messages.map((m) =>
                                    m.id === assistantMsg!.id ? assistantMsg! : m
                                );
                                onAssistantMessage({ ...assistantMsg });
                               
                                // Check if this is the final message
                                if (data.raw?.done || data.raw?.text) {
                                    meta = {
                                        model: data.model,
                                        time: data.raw?.total_duration || data.raw?.time
                                    };
                                    assistantMsg = { ...assistantMsg, content: replyBuffer, meta };
                                    chatState.messages = chatState.messages.map((m) =>
                                        m.id === assistantMsg!.id ? assistantMsg! : m
                                    );
                                    onAssistantMessage({ ...assistantMsg });
                                    onFinal && onFinal(meta);
                                }
                            } catch (e) {
                                console.error('Failed to parse SSE chunk', e);
                                chatState.errorMessage = 'Failed to parse server response';
                            }
                        }
                    });
                }
            }
        } catch (error) {
            // Check if this was an abort
            if (error instanceof DOMException && error.name === 'AbortError') {
                console.log('Stream was aborted by user');
                onStopped && onStopped();
                return;
            }
            console.error('Error reading stream:', error);
            chatState.errorMessage = 'Error reading response stream';
        } finally {
            reader.releaseLock();
            // Always reset stream state when done
            chatState.isStreamActive = false;
            currentAbortController = null;
        }
    }).catch((error) => {
        // Check if this was an abort
        if (error.name === 'AbortError') {
            console.log('Request was aborted by user');
            onStopped && onStopped();
        } else {
            console.error('Error in chat stream:', error);
            chatState.errorMessage = error.message || 'Failed to connect to chat stream';
        }
        // Always reset stream state on error/abort
        chatState.isStreamActive = false;
        currentAbortController = null;
    });
}

// Function to stop the current stream
export function stopCurrentStream(): boolean {
    if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
        chatState.isStreamActive = false;
        return true; 
    }
    return false; 
}

export function isStreamActive(): boolean {
    return chatState.isStreamActive;
}

export function setSelectedModel(model: string) {
	chatState.selectedModel = model;
}

export function fetchConversations() {
	return loadConversations();
}

export function selectConversation(conversationId: string) {
	return loadConversation(conversationId);
}

export function currentConversationId() {
	return chatState.activeConversationId;
}