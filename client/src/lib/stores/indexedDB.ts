import type { Message, Conversation } from './chatStore.svelte';
import api from '../api/api';

interface DBConversation extends Conversation {
	messages?: Message[];
}

interface DBSchema {
	conversations: DBConversation;
	messages: Message & { conversation_id: string };
	user_data: {
		id: string;
		data: any;
		updated_at: string;
	};
}

class IndexedDBManager {
	private dbName = 'nova_chat_db';
	private version = 1;
	private db: IDBDatabase | null = null;

	async init(): Promise<void> {
		return new Promise((resolve, reject) => {
			const request = indexedDB.open(this.dbName, this.version);

			request.onerror = () => reject(request.error);
			request.onsuccess = () => {
				this.db = request.result;
				resolve();
			};

			request.onupgradeneeded = (event) => {
				const db = (event.target as IDBOpenDBRequest).result;

				// Conversations store
				if (!db.objectStoreNames.contains('conversations')) {
					const conversationStore = db.createObjectStore('conversations', { keyPath: 'id' });
					conversationStore.createIndex('updatedAt', 'updatedAt', { unique: false });
				}

				// Messages store
				if (!db.objectStoreNames.contains('messages')) {
					const messageStore = db.createObjectStore('messages', { keyPath: 'id' });
					messageStore.createIndex('conversation_id', 'conversation_id', { unique: false });
					messageStore.createIndex('timestamp', 'timestamp', { unique: false });
				}

				// User data store
				if (!db.objectStoreNames.contains('user_data')) {
					db.createObjectStore('user_data', { keyPath: 'id' });
				}
			};
		});
	}

	private ensureDB(): IDBDatabase {
		if (!this.db) {
			throw new Error('Database not initialized. Call init() first.');
		}
		return this.db;
	}

	// Conversation operations
	async saveConversation(conversation: Conversation): Promise<void> {
		const db = this.ensureDB();
		return new Promise((resolve, reject) => {
			const transaction = db.transaction(['conversations'], 'readwrite');
			const store = transaction.objectStore('conversations');
			
			const request = store.put(conversation);
			request.onsuccess = () => resolve();
			request.onerror = () => reject(request.error);
		});
	}

	async saveConversations(conversations: Conversation[]): Promise<void> {
		const db = this.ensureDB();
		return new Promise((resolve, reject) => {
			const transaction = db.transaction(['conversations'], 'readwrite');
			const store = transaction.objectStore('conversations');
			
			let completed = 0;
			const total = conversations.length;

			if (total === 0) {
				resolve();
				return;
			}

			conversations.forEach(conversation => {
				const request = store.put(conversation);
				request.onsuccess = () => {
					completed++;
					if (completed === total) resolve();
				};
				request.onerror = () => reject(request.error);
			});
		});
	}

	async getConversation(id: string): Promise<Conversation | null> {
		const db = this.ensureDB();
		return new Promise((resolve, reject) => {
			const transaction = db.transaction(['conversations'], 'readonly');
			const store = transaction.objectStore('conversations');
			
			const request = store.get(id);
			request.onsuccess = () => resolve(request.result || null);
			request.onerror = () => reject(request.error);
		});
	}

	async getAllConversations(): Promise<Conversation[]> {
		const db = this.ensureDB();
		return new Promise((resolve, reject) => {
			const transaction = db.transaction(['conversations'], 'readonly');
			const store = transaction.objectStore('conversations');
			const index = store.index('updatedAt');
			
			const request = index.openCursor(null, 'prev'); // Most recent first
			const conversations: Conversation[] = [];
			
			request.onsuccess = (event) => {
				const cursor = (event.target as IDBRequest).result;
				if (cursor) {
					conversations.push(cursor.value);
					cursor.continue();
				} else {
					resolve(conversations);
				}
			};
			request.onerror = () => reject(request.error);
		});
	}

	async getRecentConversations(limit: number = 5): Promise<Conversation[]> {
		const db = this.ensureDB();
		return new Promise((resolve, reject) => {
			const transaction = db.transaction(['conversations'], 'readonly');
			const store = transaction.objectStore('conversations');
			const index = store.index('updatedAt');
			
			const request = index.openCursor(null, 'prev'); // Most recent first
			const conversations: Conversation[] = [];
			let count = 0;
			
			request.onsuccess = (event) => {
				const cursor = (event.target as IDBRequest).result;
				if (cursor && count < limit) {
					conversations.push(cursor.value);
					count++;
					cursor.continue();
				} else {
					resolve(conversations);
				}
			};
			request.onerror = () => reject(request.error);
		});
	}

	async deleteConversation(id: string): Promise<void> {
		const db = this.ensureDB();
		return new Promise((resolve, reject) => {
			const transaction = db.transaction(['conversations', 'messages'], 'readwrite');
			
			// Delete conversation
			const conversationStore = transaction.objectStore('conversations');
			const deleteConvRequest = conversationStore.delete(id);
			
			// Delete associated messages
			const messageStore = transaction.objectStore('messages');
			const messageIndex = messageStore.index('conversation_id');
			const deleteMessagesRequest = messageIndex.openCursor(IDBKeyRange.only(id));
			
			deleteMessagesRequest.onsuccess = (event) => {
				const cursor = (event.target as IDBRequest).result;
				if (cursor) {
					cursor.delete();
					cursor.continue();
				}
			};

			transaction.oncomplete = () => resolve();
			transaction.onerror = () => reject(transaction.error);
		});
	}

	// Message operations
	async saveMessage(message: Message & { conversation_id: string }): Promise<void> {
		const db = this.ensureDB();
		return new Promise((resolve, reject) => {
			const transaction = db.transaction(['messages'], 'readwrite');
			const store = transaction.objectStore('messages');
			
			const request = store.put(message);
			request.onsuccess = () => resolve();
			request.onerror = () => reject(request.error);
		});
	}

	async saveMessages(messages: (Message & { conversation_id: string })[]): Promise<void> {
		const db = this.ensureDB();
		return new Promise((resolve, reject) => {
			const transaction = db.transaction(['messages'], 'readwrite');
			const store = transaction.objectStore('messages');
			
			let completed = 0;
			const total = messages.length;

			if (total === 0) {
				resolve();
				return;
			}

			messages.forEach(message => {
				const request = store.put(message);
				request.onsuccess = () => {
					completed++;
					if (completed === total) resolve();
				};
				request.onerror = () => reject(request.error);
			});
		});
	}

	async getConversationMessages(conversationId: string): Promise<Message[]> {
		const db = this.ensureDB();
		return new Promise((resolve, reject) => {
			const transaction = db.transaction(['messages'], 'readonly');
			const store = transaction.objectStore('messages');
			const index = store.index('conversation_id');
			
			const request = index.getAll(conversationId);
			request.onsuccess = () => {
				const messages = request.result || [];
				// Sort by timestamp
				messages.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
				resolve(messages);
			};
			request.onerror = () => reject(request.error);
		});
	}

	async hasConversationMessages(conversationId: string): Promise<boolean> {
		const db = this.ensureDB();
		return new Promise((resolve, reject) => {
			const transaction = db.transaction(['messages'], 'readonly');
			const store = transaction.objectStore('messages');
			const index = store.index('conversation_id');
			
			const request = index.count(conversationId);
			request.onsuccess = () => resolve(request.result > 0);
			request.onerror = () => reject(request.error);
		});
	}

	// User data operations
	async saveUserData(id: string, data: any): Promise<void> {
		const db = this.ensureDB();
		return new Promise((resolve, reject) => {
			const transaction = db.transaction(['user_data'], 'readwrite');
			const store = transaction.objectStore('user_data');
			
			const userData = {
				id,
				data,
				updated_at: new Date().toISOString()
			};
			
			const request = store.put(userData);
			request.onsuccess = () => resolve();
			request.onerror = () => reject(request.error);
		});
	}

	async getUserData(id: string): Promise<any | null> {
		const db = this.ensureDB();
		return new Promise((resolve, reject) => {
			const transaction = db.transaction(['user_data'], 'readonly');
			const store = transaction.objectStore('user_data');
			
			const request = store.get(id);
			request.onsuccess = () => {
				const result = request.result;
				resolve(result ? result.data : null);
			};
			request.onerror = () => reject(request.error);
		});
	}

	// Utility methods
	async clearAll(): Promise<void> {
		const db = this.ensureDB();
		return new Promise((resolve, reject) => {
			const transaction = db.transaction(['conversations', 'messages', 'user_data'], 'readwrite');
			
			const stores = ['conversations', 'messages', 'user_data'];
			let completed = 0;
			
			stores.forEach(storeName => {
				const store = transaction.objectStore(storeName);
				const request = store.clear();
				request.onsuccess = () => {
					completed++;
					if (completed === stores.length) resolve();
				};
				request.onerror = () => reject(request.error);
			});
		});
	}

	async getStorageSize(): Promise<{ conversations: number; messages: number; userData: number }> {
		const db = this.ensureDB();
		return new Promise((resolve, reject) => {
			const transaction = db.transaction(['conversations', 'messages', 'user_data'], 'readonly');
			const result = { conversations: 0, messages: 0, userData: 0 };
			let completed = 0;
			
			// Count conversations
			const convStore = transaction.objectStore('conversations');
			const convRequest = convStore.count();
			convRequest.onsuccess = () => {
				result.conversations = convRequest.result;
				completed++;
				if (completed === 3) resolve(result);
			};
			
			// Count messages
			const msgStore = transaction.objectStore('messages');
			const msgRequest = msgStore.count();
			msgRequest.onsuccess = () => {
				result.messages = msgRequest.result;
				completed++;
				if (completed === 3) resolve(result);
			};
			
			// Count user data
			const userStore = transaction.objectStore('user_data');
			const userRequest = userStore.count();
			userRequest.onsuccess = () => {
				result.userData = userRequest.result;
				completed++;
				if (completed === 3) resolve(result);
			};
			
			transaction.onerror = () => reject(transaction.error);
		});
	}

	// Utility method to fetch conversation messages from API
	async fetchConversationMessagesFromAPI(conversationId: string): Promise<Message[]> {
		try {
			const response = await api.get(`/conversations/messages/${conversationId}`);
			const apiMessages = response.data.messages || [];
			
			// Transform API messages to match our Message type
			return apiMessages.map((msg: any) => ({
				id: msg.id || crypto.randomUUID?.() || Math.random().toString(36),
				role: msg.role,
				content: msg.content,
				timestamp: msg.created_at || msg.timestamp || new Date().toISOString(),
				meta: msg.meta
			}));
		} catch (error) {
			console.error('Failed to fetch messages from API:', error);
			throw error;
		}
	}

	// Method to sync missing conversation data
	async syncConversationMessages(conversationIds: string[]): Promise<void> {
		for (const conversationId of conversationIds) {
			try {
				const hasMessages = await this.hasConversationMessages(conversationId);
				if (!hasMessages) {
					const messages = await this.fetchConversationMessagesFromAPI(conversationId);
					const messagesWithConvId = messages.map(msg => ({
						...msg,
						conversation_id: conversationId
					}));
					
					if (messagesWithConvId.length > 0) {
						await this.saveMessages(messagesWithConvId);
					}
				}
			} catch (error) {
				console.error(`Failed to sync messages for conversation ${conversationId}:`, error);
				// Continue with other conversations even if one fails
			}
		}
	}
}

// Export singleton instance
export const indexedDBManager = new IndexedDBManager();

// Initialize on module load
let initPromise: Promise<void> | null = null;

export async function ensureDBInitialized(): Promise<void> {
	if (!initPromise) {
		initPromise = indexedDBManager.init();
	}
	return initPromise;
}