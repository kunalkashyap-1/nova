// Model types
export interface Model {
  id: number;
  name: string;
  provider: string;
  model_id: string;
  description?: string;
  capabilities?: Record<string, boolean>;
  parameters?: Record<string, any>;
  is_active: boolean;
  cost_per_1k_input_tokens?: number;
  cost_per_1k_output_tokens?: number;
  max_tokens?: number;
}

export interface ModelParameter {
  name: string;
  default_value?: number;
  min_value?: number;
  max_value?: number;
  step?: number;
}

// Message types
export interface Message {
  id: number;
  conversation_id: number;
  role: 'user' | 'assistant' | 'system';
  content: string;
  tokens_used?: number;
  model_id?: number;
  metadata?: Record<string, any>;
  parent_message_id?: number;
  created_at: string;
  updated_at: string;
}

export interface Conversation {
  id: number;
  title: string;
  user_id: number;
  model_id?: number;
  system_prompt?: string;
  folder_id?: number;
  is_pinned: boolean;
  is_archived: boolean;
  last_message_at?: string;
  created_at: string;
  updated_at: string;
}

// Request/Response types
export interface MessageRequest {
  chat_id: number;
  user_id: number;
  model: string;
  message: string;
  provider: 'ollama' | 'huggingface';
  stream?: boolean;
  context_strategy?: 'hybrid' | 'vectordb' | 'cache' | 'web_search';
  optimize_context?: boolean;
  max_context_docs?: number;
  web_search?: boolean;
  web_search_query?: string;
  max_search_results?: number;
}

export interface MessageResponse {
  model: string;
  user_id: number;
  chat_id: number;
  reply: string;
  raw?: any;
  error?: string;
  status_code?: number;
}

export interface ErrorResponse {
  error: string;
  message?: string;
  status_code: number;
  request_id?: string;
}

// User types
export interface User {
  id: number;
  username: string;
  email: string;
  full_name: string;
  profile_picture?: string;
  bio?: string;
  created_at: string;
}

// UI types
export interface TabItem {
  id: string;
  label: string;
  icon?: string;
}

export interface MenuItem {
  id: string;
  label: string;
  icon?: string;
  action?: () => void;
  subitems?: MenuItem[];
}
