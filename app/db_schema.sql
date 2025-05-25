CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    profile_picture TEXT DEFAULT NULL,
    bio TEXT DEFAULT '',
    preferred_language VARCHAR(50) DEFAULT '',
    timezone VARCHAR(100) DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Models table to store different AI models available
CREATE TABLE models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    provider VARCHAR(100) NOT NULL, -- e.g., OpenAI, Anthropic, etc.
    model_id VARCHAR(100) NOT NULL, -- The ID used by the provider API
    description TEXT,
    capabilities JSONB, -- Store capabilities as JSON (e.g., image generation, code completion)
    parameters JSONB, -- Default parameters as JSON
    is_active BOOLEAN DEFAULT true,
    cost_per_1k_input_tokens NUMERIC(10, 6),
    cost_per_1k_output_tokens NUMERIC(10, 6),
    max_tokens INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_model_provider_id UNIQUE (provider, model_id)
);

-- 5 Best Ollama Models
INSERT INTO models (
    name, provider, model_id, description, capabilities, parameters, is_active, cost_per_1k_input_tokens, cost_per_1k_output_tokens, max_tokens
) VALUES
('Llama 3 8B Q4', 'ollama', 'llama3:8b-q4', 'Meta Llama 3, 8B parameters, quantized 4-bit', '{"chat": true, "open_source": true}', '{"quantization": "q4", "size": "8B"}', true, 0, 0, 8192),
('Mistral 7B Q4', 'ollama', 'mistral:7b-q4', 'Mistral AI, 7B parameters, quantized 4-bit', '{"chat": true, "open_source": true}', '{"quantization": "q4", "size": "7B"}', true, 0, 0, 8192),
('Phi-3 Mini 3.8B Q4', 'ollama', 'phi3:3.8b-q4', 'Microsoft Phi-3 Mini, 3.8B parameters, quantized 4-bit', '{"chat": true, "open_source": true}', '{"quantization": "q4", "size": "3.8B"}', true, 0, 0, 4096),
('Gemma 2B Q4', 'ollama', 'gemma:2b-q4', 'Google Gemma, 2B parameters, quantized 4-bit', '{"chat": true, "open_source": true}', '{"quantization": "q4", "size": "2B"}', true, 0, 0, 4096),
('Qwen2 4B Q4', 'ollama', 'qwen2:4b-q4', 'Alibaba Qwen2, 4B parameters, quantized 4-bit', '{"chat": true, "open_source": true}', '{"quantization": "q4", "size": "4B"}', true, 0, 0, 8192);

-- 5 Top Hugging Face Leaderboard Models (no Gemma, last 3 are HF exclusive)
INSERT INTO models (
    name, provider, model_id, description, capabilities, parameters, is_active, cost_per_1k_input_tokens, cost_per_1k_output_tokens, max_tokens
) VALUES
('Qwen2.5-3B-Instruct', 'huggingface', 'Qwen/Qwen2.5-3B-Instruct', 'Qwen2.5 3B Instruct, top leaderboard model', '{"chat": true, "open_source": true}', '{"size": "3B"}', true, 0, 0, 8192),
('Llama-3.2-3B-Instruct', 'huggingface', 'meta-llama/Llama-3.2-3B-Instruct', 'Meta Llama 3.2 3B Instruct, top leaderboard model', '{"chat": true, "open_source": true}', '{"size": "3B"}', true, 0, 0, 8192),
('Flan-T5 Large', 'huggingface', 'google/flan-t5-large', 'Google Flan-T5 Large, top leaderboard model, HF exclusive', '{"chat": true, "open_source": true}', '{"size": "1B"}', true, 0, 0, 2048),
('CoolQwen-3B-IT', 'huggingface', 'ehristoforu/coolqwen-3b-it', 'CoolQwen 3B IT, top leaderboard model, HF exclusive', '{"chat": true, "open_source": true}', '{"size": "3B"}', true, 0, 0, 8192),
('Qwen2.5-0.5B-Instruct', 'huggingface', 'FlofloB/100k_fineweb_continued_pretraining_Qwen2.5-0.5B-Instruct_Unsloth_merged_16bit', 'Qwen2.5-0.5B-Instruct, top leaderboard model, HF exclusive', '{"chat": true, "open_source": true}', '{"size": "0.5B"}', true, 0, 0, 2048);

-- Conversations table
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL DEFAULT 'New Conversation',
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    model_id INTEGER REFERENCES models(id) ON DELETE SET NULL,
    system_prompt TEXT,
    folder_id INTEGER,
    is_pinned BOOLEAN DEFAULT false,
    is_archived BOOLEAN DEFAULT false,
    last_message_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Messages table for storing conversation messages
CREATE TABLE messages (
    id SERIAL PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL, -- 'user', 'assistant', or 'system'
    content TEXT NOT NULL,
    tokens_used INTEGER, -- Count of tokens used for billing
    model_id INTEGER REFERENCES models(id) ON DELETE SET NULL, -- The model that generated this message
    message_metadata JSONB, -- For storing additional data like citations, code blocks, etc.
    parent_message_id INTEGER REFERENCES messages(id) ON DELETE SET NULL, -- For threaded conversations
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- For organizing conversations
CREATE TABLE folders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    color VARCHAR(50), -- For UI customization
    parent_folder_id INTEGER REFERENCES folders(id) ON DELETE CASCADE, -- For nested folders
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for storing conversation tags
CREATE TABLE tags (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    color VARCHAR(50), -- For UI customization
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE, -- Optional: if tags are user-specific
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Junction table for conversation-tag many-to-many relationship
CREATE TABLE conversation_tags (
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (conversation_id, tag_id)
);

-- For storing user-specific model preferences
CREATE TABLE user_model_preferences (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    custom_parameters JSONB, -- User-specific settings for a model
    is_favorite BOOLEAN DEFAULT false,
    PRIMARY KEY (user_id, model_id)
);

-- File attachments for conversations
CREATE TABLE attachments (
    id SERIAL PRIMARY KEY,
    message_id INTEGER REFERENCES messages(id) ON DELETE CASCADE,
    file_name VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_type VARCHAR(100) NOT NULL,
    file_size INTEGER NOT NULL, -- Size in bytes
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Addition triggers and procedures
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
   NEW.updated_at = NOW();
   RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for each table to automatically update updated_at columns
CREATE TRIGGER trigger_users_updated_at
BEFORE UPDATE ON users
FOR EACH ROW
EXECUTE PROCEDURE update_updated_at_column();

CREATE TRIGGER trigger_models_updated_at
BEFORE UPDATE ON models
FOR EACH ROW
EXECUTE PROCEDURE update_updated_at_column();

CREATE TRIGGER trigger_conversations_updated_at
BEFORE UPDATE ON conversations
FOR EACH ROW
EXECUTE PROCEDURE update_updated_at_column();

CREATE TRIGGER trigger_messages_updated_at
BEFORE UPDATE ON messages
FOR EACH ROW
EXECUTE PROCEDURE update_updated_at_column();

CREATE TRIGGER trigger_folders_updated_at
BEFORE UPDATE ON folders
FOR EACH ROW
EXECUTE PROCEDURE update_updated_at_column();

-- Function and trigger to update last_message_at in conversations when a message is added
CREATE OR REPLACE FUNCTION update_conversation_last_message_time()
RETURNS TRIGGER AS $$
BEGIN
   UPDATE conversations
   SET last_message_at = NEW.created_at
   WHERE id = NEW.conversation_id;
   RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER trigger_update_conversation_last_message_time
AFTER INSERT ON messages
FOR EACH ROW
EXECUTE PROCEDURE update_conversation_last_message_time();
