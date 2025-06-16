import { writable, derived, get } from 'svelte/store';
import type { Model } from '$lib/types';
import { browser } from '$app/environment';
import api from '../api/api';
import { setSelectedModel } from './chatStore.svelte';

// This is now the single source of truth for model state. All components should use this store.

// Default model parameters
const DEFAULT_PARAMS = {
	temperature: 0.7,
	top_p: 0.9,
	max_tokens: 1000
};

// Define store state type for reuse
type ModelStoreState = {
	models: Model[];
	loading: boolean;
	error: string | null;
	selectedModelId: number | null;
	selectedModelName: string | null;
	selectedProvider: string;
	parameters: Record<string, any>;
};

// Initialize stores
const createModelStore = () => {
	// Models list cache
	const store = writable<ModelStoreState>({
		models: [],
		loading: false,
		error: null,
		selectedModelId: null,
		selectedModelName: 'deepseek-r1:1.5b',
		selectedProvider: 'ollama',
		parameters: { ...DEFAULT_PARAMS }
	});

	// Load from localStorage if available
	if (browser) {
		const savedState = localStorage.getItem('nova_model_preferences');
		if (savedState) {
			try {
				const parsed = JSON.parse(savedState);
				store.update((state: ModelStoreState) => ({
					...state,
					selectedModelId: parsed.selectedModelId || null,
					selectedModelName: parsed.selectedModelName || 'deepseek-r1:1.5b',
					selectedProvider: parsed.selectedProvider || 'ollama',
					parameters: { ...DEFAULT_PARAMS, ...parsed.parameters }
				}));
				setSelectedModel(parsed.selectedModelName);
			} catch (e) {
				console.error('Failed to parse saved model preferences', e);
			}
		}
	}

	return {
		subscribe: store.subscribe,

		// Load all available models
		fetchModels: async () => {
			store.update((state: ModelStoreState) => ({ ...state, loading: true, error: null }));
			try {
				const response = await api.get('/models/?provider=all');
				const models = response.data;
				store.update((state: ModelStoreState) => ({
					...state,
					models,
					loading: false,
					selectedModelId: state.selectedModelId || (models.length > 0 ? models[0].id : null)
				}));
			} catch (error) {
				console.error('Error fetching models:', error);
				store.update((state: ModelStoreState) => ({
					...state,
					loading: false,
					error: error instanceof Error ? error.message : 'Unknown error'
				}));
			}
		},

		// Fetch models for a specific provider
		fetchModelsByProvider: async (provider: string) => {
			store.update((state: ModelStoreState) => ({ ...state, loading: true, error: null }));
			try {
				const response = await api.get(`/models?provider=${provider}`);
				const models = response.data;
				store.update((state: ModelStoreState) => ({
					...state,
					models,
					loading: false,
					selectedProvider: provider
				}));
				if (browser) {
					const currentState = get(store) as ModelStoreState;
					localStorage.setItem(
						'nova_model_preferences',
						JSON.stringify({
							selectedModelId: currentState.selectedModelId,
							selectedProvider: provider,
							selectedModelName: currentState.selectedModelName,
							parameters: currentState.parameters
						})
					);
				}
			} catch (error) {
				console.error(`Error fetching ${provider} models:`, error);
				store.update((state: ModelStoreState) => ({
					...state,
					loading: false,
					error: error instanceof Error ? error.message : 'Unknown error'
				}));
			}
		},

		// Select a model
		selectModel: (modelId: number) => {
			store.update((state: ModelStoreState) => {
				// Find the model to get its default parameters
				const model = state.models.find((m: Model) => m.id === modelId);
				let newParams = { ...state.parameters };

				// If model has its own parameters, use them as base
				if (model?.parameters) {
					newParams = { ...DEFAULT_PARAMS, ...model.parameters };
				}

				// Update chatStore with model name
				if (model) {
					setSelectedModel(model.model_id);
				}

				// Save preference
				if (browser) {
					localStorage.setItem(
						'nova_model_preferences',
						JSON.stringify({
							selectedModelId: modelId,
							selectedProvider: model?.provider,
							selectedModelName: model?.model_id,
							parameters: newParams
						})
					);
				}

				return {
					...state,
					selectedModelId: modelId,
					parameters: newParams
				};
			});
		},

		// Update model parameters
		updateParameter: (name: string, value: any) => {
			store.update((state: ModelStoreState) => {
				const newParams = { ...state.parameters, [name]: value };

				// Save preference
				if (browser) {
					const currentState = get(store) as ModelStoreState;
					localStorage.setItem(
						'nova_model_preferences',
						JSON.stringify({
							selectedModelId: currentState.selectedModelId,
							selectedProvider: currentState.selectedProvider,
							selectedModelName: currentState.selectedModelName,
							parameters: newParams
						})
					);
				}

				return { ...state, parameters: newParams };
			});
		},

		// Helper method to get current state
		get: () => {
			return get(store);
		}
	};
};

// Create the store
export const modelStore = createModelStore();

// Create a derived store for the currently selected model
export const selectedModel = derived<typeof modelStore, Model | null>(
	modelStore,
	($modelStore: ModelStoreState) =>
		$modelStore.models.find((m: Model) => m.id === $modelStore.selectedModelId) || null
);

// Create derived stores for convenience
export const modelParameters = derived<typeof modelStore, Record<string, any>>(
	modelStore,
	($modelStore: ModelStoreState) => $modelStore.parameters
);

export const availableModels = derived<typeof modelStore, Model[]>(
	modelStore,
	($modelStore: ModelStoreState) => $modelStore.models
);

export const isLoading = derived<typeof modelStore, boolean>(
	modelStore,
	($modelStore: ModelStoreState) => $modelStore.loading
);

export const errorMessage = derived<typeof modelStore, string | null>(
	modelStore,
	($modelStore: ModelStoreState) => $modelStore.error
);
