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
	selectedModel: 'llama3.2:1b'
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
				isGuest: !parsed.username
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
	const user_id = authUser.username || 'Guest';
	const model = chatState.selectedModel || 'llama3.2:1b';

	// Only add user message if this is not the first message (for follow-ups)
	if (
		chatState.messages.length === 0 ||
		chatState.messages[chatState.messages.length - 1].role !== 'user' ||
		chatState.messages[chatState.messages.length - 1].content !== userMessage
	) {
		const userMsg: Message = {
			id: uuidv4(),
			role: 'user',
			content: userMessage,
			timestamp: new Date().toISOString()
		};
		chatState.messages = [...chatState.messages, userMsg];
	}

	let assistantMsg: Message | null = null;
	let replyBuffer = '';
	let meta: { model?: string; time?: number } = {};

	fetch(`${import.meta.env.VITE_BACKEND_API_URL}/api/v1/messages/`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({
			user_id,
			model,
			message: userMessage,
			provider: 'ollama',
			chat_id: chatId
		})
	}).then(async (response) => {
		if (!response.body) return;
		const reader = response.body.getReader();
		const decoder = new TextDecoder();
		let done = false;
		while (!done) {
			const { value, done: doneReading } = await reader.read();
			done = doneReading;
			if (value) {
				const chunk = decoder.decode(value, { stream: true });
				chunk.split(/\n\n/).forEach((eventStr) => {
					if (eventStr.startsWith('data: ')) {
						try {
							const data = JSON.parse(eventStr.slice(6));
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
							if (data.raw?.done) {
								meta = { model: data.model, time: data.raw.total_duration };
								assistantMsg = { ...assistantMsg, content: replyBuffer, meta };
								chatState.messages = chatState.messages.map((m) =>
									m.id === assistantMsg!.id ? assistantMsg! : m
								);
								onAssistantMessage({ ...assistantMsg });
								onFinal && onFinal(meta);
							}
						} catch (e) {
							console.error('Failed to parse SSE chunk', e);
						}
					}
				});
			}
		}
	});
}

export function setSelectedModel(model: string) {
	chatState.selectedModel = model;
}
