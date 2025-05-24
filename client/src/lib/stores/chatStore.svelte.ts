import { v4 as uuidv4 } from 'uuid';
import api from '../api';
// This is now the single source of truth for chat/conversation state. All components should use this store.
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
	selectedModel: ''
});

// --- User Auth State ---
export type AuthUser = {
	username: string;
	profilePicUrl?: string;
	isGuest: boolean;
	tempUserId?: number;
};

function generateTempUserId(): number {
	// Generate a random number between 10000 and 99999
	return Math.floor(Math.random() * 90000) + 10000;
}

function loadUserFromLocalStorage(): AuthUser {
	try {
		const raw = localStorage.getItem('nova_user');
		if (raw) {
			const parsed = JSON.parse(raw);
			return {
				username: parsed.username || 'Guest User',
				profilePicUrl: parsed.profilePicUrl || '',
				isGuest: !parsed.username,
				tempUserId: parsed.tempUserId || generateTempUserId()
			};
		}
	} catch {}
	// For new users, generate a temp ID
	const tempUserId = generateTempUserId();
	return { 
		username: 'Guest User', 
		profilePicUrl: '', 
		isGuest: true,
		tempUserId 
	};
}

export const authUser: AuthUser = $state(loadUserFromLocalStorage());

export function setUser(user: Partial<AuthUser>) {
	const merged = { 
		...authUser, 
		...user, 
		isGuest: !user.username,
		tempUserId: authUser.tempUserId || generateTempUserId() // Preserve or generate temp ID
	};
	authUser.username = merged.username;
	authUser.profilePicUrl = merged.profilePicUrl || '';
	authUser.isGuest = merged.isGuest;
	authUser.tempUserId = merged.tempUserId;
	localStorage.setItem('nova_user', JSON.stringify(merged));
}

export function signOut() {
	authUser.username = 'Guest User';
	authUser.profilePicUrl = '';
	authUser.isGuest = true;
	localStorage.removeItem('nova_user');
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

export async function createNewChat(prompt: string): Promise<string | undefined> {
	if (!prompt.trim()) {
		setError('Please enter a chat prompt');
		return undefined;
	}
	setLoading(true);
	try {
		const newId = uuidv4();
		chatState.activeConversationId = newId;
		// Add conversation to list
		const newConversation: Conversation = {
			id: newId,
			title: `Chat: ${prompt.slice(0, 30)}${prompt.length > 30 ? '...' : ''}`,
			updatedAt: new Date().toISOString()
		};
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
	onFinal?: (meta: { model?: string; time?: number }) => void
) {
	const user_id = authUser.isGuest ? authUser.tempUserId : parseInt(authUser.username);
	const model = chatState.selectedModel || 'llama3.2:1b';

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
			provider: 'ollama',
			stream: true,
			context_strategy: 'hybrid',
			optimize_context: true,
			max_context_docs: 15
		})
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
			console.error('Error reading stream:', error);
			chatState.errorMessage = 'Error reading response stream';
		} finally {
			reader.releaseLock();
		}
	}).catch((error) => {
		console.error('Error in chat stream:', error);
		chatState.errorMessage = error.message || 'Failed to connect to chat stream';
	});
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
