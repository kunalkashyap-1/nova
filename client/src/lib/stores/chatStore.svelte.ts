import { v4 as uuidv4 } from 'uuid';
import api from '../api/api';
import { authStore } from './auth.svelte';
import { indexedDBManager, ensureDBInitialized } from './indexedDB';

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
	isStreamActive: boolean;
	isDBInitialized: boolean;
	webSearchEnabled: boolean; // whether to request web results
};

export const chatState: ChatState = $state({
	conversations: [],
	messages: [],
	activeConversationId: null,
	isLoading: false,
	errorMessage: '',
	selectedModel: 'deepseek-r1:1.5b',
	isStreamActive: false,
	isDBInitialized: false,
    webSearchEnabled: false
});

export function setLoading(isLoading: boolean) {
	chatState.isLoading = isLoading;
}

export function toggleWebSearch() {
    chatState.webSearchEnabled = !chatState.webSearchEnabled;
}

export function setError(message: string) {
	chatState.errorMessage = message;
	setTimeout(() => {
		if (chatState.errorMessage === message) {
			chatState.errorMessage = '';
		}
	}, 5000);
}

// Initialize IndexedDB and sync recent conversations
export async function initializeDB(): Promise<void> {
	try {
		await ensureDBInitialized();
		chatState.isDBInitialized = true;
		
		// Load conversations from IndexedDB first
		const localConversations = await indexedDBManager.getAllConversations();
		chatState.conversations = localConversations;
		
		// Get the 5 most recent conversations and check if we have their messages
		const recentConversations = await indexedDBManager.getRecentConversations(5);
		const conversationsToFetch: string[] = [];
		
		for (const conv of recentConversations) {
            // Only sync remote (numeric) conversations
            if (!/^[0-9]+$/.test(conv.id)) continue;
            const hasMessages = await indexedDBManager.hasConversationMessages(conv.id);
            if (!hasMessages) {
                conversationsToFetch.push(conv.id);
            }
        }
		
		// Fetch missing conversation messages from API
		if (conversationsToFetch.length > 0) {
			await fetchAndSyncConversationMessages(conversationsToFetch);
		}
		
	} catch (error) {
		console.error('Failed to initialize IndexedDB:', error);
		setError('Failed to initialize local storage');
		// Fallback to localStorage
		chatState.conversations = loadConversationsFromStorage();
	}
}

// Fetch messages for conversations that don't have them locally
async function fetchAndSyncConversationMessages(conversationIds: string[]): Promise<void> {
	try {
		if (chatState.isDBInitialized) {
			await indexedDBManager.syncConversationMessages(conversationIds);
		}
	} catch (error) {
		console.error('Failed to sync conversation messages:', error);
	}
}

// --- Conversation Persistence (Updated) ---
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

// Updated to use IndexedDB
async function saveConversationsToIndexedDB(conversations: Conversation[]) {
    // Prepare plain objects to avoid proxies / non-cloneable values
    const plainConversations: Conversation[] = conversations.map((c) => ({
        id: c.id,
        title: c.title,
        updatedAt: c.updatedAt || new Date().toISOString()
    }));
	if (chatState.isDBInitialized) {
		try {
			await indexedDBManager.saveConversations(plainConversations);
		} catch (error) {
			console.error('Failed to save conversations to IndexedDB:', error);
			// Fallback to localStorage
			saveConversationsToStorage(conversations);
		}
	} else {
		saveConversationsToStorage(conversations);
	}
}

export async function loadConversations() {
    // Ensure IndexedDB is ready before attempting to read
    if (!chatState.isDBInitialized) {
        try {
            await initializeDB();
        } catch (e) {
            console.error('Failed to initialize DB in loadConversations:', e);
        }
    }
	setLoading(true);
	try {
        if (chatState.isDBInitialized) {
            // 1️⃣ Load from IndexedDB first for instant UI feedback.
            chatState.conversations = await indexedDBManager.getAllConversations();
        } else {
            chatState.conversations = loadConversationsFromStorage();
        }

        // 2️⃣ Always attempt to fetch the latest list from the backend so that
        //    we have conversations created on other devices or after clearing storage.
        try {
            const { data } = await api.get('/conversations/');
            if (Array.isArray(data)) {
                const remoteConversations: Conversation[] = data.map((c: any) => ({
                    id: String(c.id ?? c.uuid ?? c.conversation_id ?? crypto.randomUUID?.() ?? Math.random().toString(36)),
                    title: c.title ?? 'Untitled',
                    // fall back to created_at if updated_at missing
                    updatedAt: c.updated_at ?? c.updatedAt ?? c.created_at ?? new Date().toISOString(),
                }));

                // Merge remote with local (keep most recent updatedAt for duplicates)
                const mergedMap = new Map<string, Conversation>();
                [...chatState.conversations, ...remoteConversations].forEach((conv) => {
                    const existing = mergedMap.get(conv.id);
                    if (!existing || new Date(conv.updatedAt).getTime() > new Date(existing.updatedAt).getTime()) {
                        mergedMap.set(conv.id, conv);
                    }
                });
                chatState.conversations = Array.from(mergedMap.values()).sort((a, b) =>
                    new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
                );

                // Persist latest copy locally for offline use
                if (chatState.isDBInitialized) {
                    await saveConversationsToIndexedDB(chatState.conversations);
                    // Sync missing messages for remote numeric conversations
                    const numericConvIds = chatState.conversations
                        .filter((c) => /^[0-9]+$/.test(c.id))
                        .map((c) => c.id);
                    const missingConvIds: string[] = [];
                    for (const cid of numericConvIds) {
                        const hasMsgs = await indexedDBManager.hasConversationMessages(cid);
                        if (!hasMsgs) missingConvIds.push(cid);
                    }
                    if (missingConvIds.length > 0) {
                        await fetchAndSyncConversationMessages(missingConvIds);
                    }
                } else {
                    saveConversationsToStorage(chatState.conversations);
                }
            }
        } catch (apiErr) {
            // If API fetch fails we still have the local conversations, so only log
            console.error('Failed to fetch conversations from API:', apiErr);
        }
    } catch (e) {
        console.error('Failed to load conversations:', e);
        setError('Failed to load conversations');
        // Fallback to localStorage
        chatState.conversations = loadConversationsFromStorage();
    } finally {
        setLoading(false);
        if (!chatState.activeConversationId && chatState.conversations.length > 0) {
            // Prefer first remote (numeric) conversation
            const initialConv = chatState.conversations.find((c) => /^[0-9]+$/.test(c.id)) || chatState.conversations[0];
            await loadConversation(initialConv.id);
        }
    }
}

export async function loadConversation(id: string) {
    // Ensure IndexedDB is ready before attempting to load messages
    if (!chatState.isDBInitialized) {
        try {
            await initializeDB();
        } catch (e) {
            console.error('Failed to initialize DB in loadConversation:', e);
        }
    }
	setLoading(true);
	
	if (chatState.activeConversationId !== id) {
		chatState.messages = [];
		
		// Load messages from IndexedDB if available
		if (chatState.isDBInitialized) {
			try {
				const messages = await indexedDBManager.getConversationMessages(id);
				chatState.messages = messages;
				
				// If no messages found locally, try to fetch from API
				if (messages.length === 0) {
					try {
						const apiMessages = await indexedDBManager.fetchConversationMessagesFromAPI(id);
						const messagesWithConvId = apiMessages.map(msg => ({
							...msg,
							conversation_id: id
						}));
						
						chatState.messages = apiMessages;
						
						// Save to IndexedDB for future use
						if (messagesWithConvId.length > 0) {
							await indexedDBManager.saveMessages(messagesWithConvId);
						}
					} catch (apiError) {
						console.error('Failed to fetch messages from API:', apiError);
					}
				}
			} catch (error) {
				console.error('Failed to load messages from IndexedDB:', error);
			}
		}
	}
	
	chatState.activeConversationId = id;
	setLoading(false);
}

// Helper function to get user ID from auth store
function getUserId(): string | number {
	const user = authStore.user;
	if (!user) {
		return 'guest';
	}
	return user.id || user.username || 'guest';
}

export async function createNewChat(prompt: string): Promise<string | undefined> {
    // Make sure DB is ready before we start saving
    if (!chatState.isDBInitialized) {
        try {
            await initializeDB();
        } catch (e) {
            console.error('Failed to initialize DB in createNewChat:', e);
        }
    }
	setLoading(true);
	try {
		const user_id = getUserId();
		
		// Create conversation on backend first
		try {
			const response = await api.post('/conversations/', {
				title: `Chat: ${prompt.slice(0, 30)}${prompt.length > 30 ? '...' : ''}`,
				user_id: user_id
			});

			const apiConv = response.data;
            const newConversation: Conversation = {
                id: apiConv.id,
                title: apiConv.title,
                updatedAt: apiConv.updated_at || apiConv.updatedAt || new Date().toISOString()
            };
			const newId = newConversation.id;
			chatState.activeConversationId = newId;
			
			// Add conversation to list and save to IndexedDB
			chatState.conversations = [newConversation, ...chatState.conversations];
			await saveConversationsToIndexedDB(chatState.conversations);
			
			// Create user message first
			const userMsg: Message = {
				id: uuidv4(),
				role: 'user',
				content: prompt,
				timestamp: new Date().toISOString()
			};
			chatState.messages = [userMsg];
			
			// Save user message to IndexedDB
			if (chatState.isDBInitialized) {
				await indexedDBManager.saveMessage({ ...userMsg, conversation_id: newId });
			}
			
			// Now connect to SSE after conversation is created
			connectToChatSSE(
				newId,
				prompt,
				(msg) => {},
				async (meta) => {
					const conversation = chatState.conversations.find((c) => c.id === newId);
					if (conversation) {
						conversation.updatedAt = new Date().toISOString();
						chatState.conversations = [...chatState.conversations];
						await saveConversationsToIndexedDB(chatState.conversations);
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
    // Ensure DB before we save conversation updates & messages
    if (!chatState.isDBInitialized) {
        try {
            await initializeDB();
        } catch (e) {
            console.error('Failed to initialize DB in sendMessage:', e);
        }
    }
	if (!content.trim() || !chatState.activeConversationId) return;
	const chatId = chatState.activeConversationId;
	
	const conversation = chatState.conversations.find((c) => c.id === chatId);
	if (conversation) {
		conversation.updatedAt = new Date().toISOString();
		chatState.conversations = [...chatState.conversations];
		await saveConversationsToIndexedDB(chatState.conversations);
	}
	
	connectToChatSSE(
		chatId,
		content,
		(msg) => {},
		async (meta) => {
			const conversation = chatState.conversations.find((c) => c.id === chatId);
			if (conversation) {
				conversation.updatedAt = new Date().toISOString();
				chatState.conversations = [...chatState.conversations];
				await saveConversationsToIndexedDB(chatState.conversations);
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
    let userMsg: Message;
    
    if (!lastMessage || lastMessage.content !== userMessage) {
        userMsg = {
            id: uuidv4(),
            role: 'user',
            content: userMessage,
            timestamp: new Date().toISOString()
        };
        chatState.messages = [...chatState.messages, userMsg];
        
        // Save user message to IndexedDB
        if (chatState.isDBInitialized) {
            indexedDBManager.saveMessage({ ...userMsg, conversation_id: chatId }).catch(console.error);
        }
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
            web_search: chatState.webSearchEnabled,
            web_search_query: chatState.webSearchEnabled ? userMessage : undefined,
            max_search_results: chatState.webSearchEnabled ? 5 : undefined,
            stream: true,
            context_strategy: chatState.webSearchEnabled ? 'web_search' : 'hybrid',
            optimize_context: true,
            max_context_docs: 15
        }),
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
                                    
                                    // Save final assistant message to IndexedDB
                                    if (chatState.isDBInitialized && assistantMsg) {
                                        indexedDBManager.saveMessage({ 
                                            ...assistantMsg, 
                                            conversation_id: chatId 
                                        }).catch(console.error);
                                    }
                                    
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

// Additional utility functions for IndexedDB management
export async function clearLocalData(): Promise<void> {
	if (chatState.isDBInitialized) {
		try {
			await indexedDBManager.clearAll();
			chatState.conversations = [];
			chatState.messages = [];
			chatState.activeConversationId = null;
		} catch (error) {
			console.error('Failed to clear IndexedDB:', error);
			setError('Failed to clear local data');
		}
	}
	// Also clear localStorage as fallback
	localStorage.removeItem('nova_conversations');
}

export async function getStorageInfo() {
	if (chatState.isDBInitialized) {
		try {
			return await indexedDBManager.getStorageSize();
		} catch (error) {
			console.error('Failed to get storage info:', error);
			return null;
		}
	}
	return null;
}