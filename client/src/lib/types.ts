export interface Model {
    id: number;
    name: string;
    provider: string;
    model_id: string;
    description?: string;
    capabilities?: {
        chat?: boolean;
        open_source?: boolean;
        [key: string]: any;
    };
    parameters?: {
        temperature?: number;
        top_p?: number;
        max_tokens?: number;
        [key: string]: any;
    };
    is_active: boolean;
    cost_per_1k_input_tokens?: number;
    cost_per_1k_output_tokens?: number;
    max_tokens?: number;
    created_at: string;
    updated_at: string;
}

export interface ModelParameters {
    temperature: number;
    top_p: number;
    max_tokens: number;
    [key: string]: any;
}

export interface ModelStoreState {
    models: Model[];
    loading: boolean;
    error: string | null;
    selectedModelId: number | null;
    selectedProvider: string;
    parameters: ModelParameters;
} 