// NVIDIA NIM Provider for Glass
// Compatible with NVIDIA NIM API (OpenAI-compatible endpoint)
// Base URL: https://integrate.api.nvidia.com/v1

class NvidiaNimProvider {
    static async validateApiKey(key) {
        if (!key || typeof key !== 'string' || !key.startsWith('nvapi-')) {
            return { success: false, error: 'Invalid NVIDIA NIM API key format. Key must start with "nvapi-".' };
        }

        try {
            // NVIDIA NIM uses OpenAI-compatible /v1/models endpoint
            const response = await fetch('https://integrate.api.nvidia.com/v1/models', {
                headers: {
                    'Authorization': `Bearer ${key}`,
                    'Content-Type': 'application/json',
                }
            });

            if (response.ok) {
                return { success: true };
            } else {
                const errorData = await response.json().catch(() => ({}));
                const message = errorData.message || errorData.error?.message || `Validation failed with status: ${response.status}`;
                return { success: false, error: message };
            }
        } catch (error) {
            console.error('[NvidiaNimProvider] Network error during key validation:', error);
            return { success: false, error: 'A network error occurred during validation.' };
        }
    }
}

/**
 * Creates an NVIDIA NIM LLM instance (non-streaming)
 * @param {object} opts
 * @param {string} opts.apiKey - NVIDIA NIM API key (starts with nvapi-)
 * @param {string} [opts.model='meta/llama-3.1-70b-instruct'] - Model ID
 * @param {number} [opts.temperature=0.7]
 * @param {number} [opts.maxTokens=2048]
 * @returns {object} LLM instance
 */
function createLLM({ apiKey, model = 'meta/llama-3.1-70b-instruct', temperature = 0.7, maxTokens = 2048 }) {
    const BASE_URL = 'https://integrate.api.nvidia.com/v1';

    const callApi = async (messages) => {
        const response = await fetch(`${BASE_URL}/chat/completions`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model,
                messages,
                temperature,
                max_tokens: maxTokens,
            }),
        });

        if (!response.ok) {
            const errBody = await response.json().catch(() => ({}));
            throw new Error(`NVIDIA NIM API error ${response.status}: ${errBody.message || response.statusText}`);
        }

        const result = await response.json();
        return {
            content: result.choices[0].message.content.trim(),
            raw: result,
        };
    };

    return {
        // Used by screenshot/ask pipeline — parts may include text strings and inlineData (images)
        generateContent: async (parts) => {
            const messages = [];
            let systemPrompt = '';
            const userContent = [];

            for (const part of parts) {
                if (typeof part === 'string') {
                    if (!systemPrompt && part.includes('You are')) {
                        systemPrompt = part;
                    } else {
                        userContent.push({ type: 'text', text: part });
                    }
                } else if (part.inlineData) {
                    // Vision-capable NIM models accept image_url with base64 data URIs
                    userContent.push({
                        type: 'image_url',
                        image_url: {
                            url: `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`,
                        },
                    });
                }
            }

            if (systemPrompt) messages.push({ role: 'system', content: systemPrompt });
            if (userContent.length > 0) messages.push({ role: 'user', content: userContent });

            const result = await callApi(messages);
            return {
                response: { text: () => result.content },
                raw: result.raw,
            };
        },

        chat: async (messages) => {
            return await callApi(messages);
        },
    };
}

/**
 * Creates an NVIDIA NIM streaming LLM instance
 */
function createStreamingLLM({ apiKey, model = 'meta/llama-3.1-70b-instruct', temperature = 0.7, maxTokens = 2048 }) {
    const BASE_URL = 'https://integrate.api.nvidia.com/v1';

    return {
        streamChat: async (messages) => {
            const response = await fetch(`${BASE_URL}/chat/completions`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${apiKey}`,
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model,
                    messages,
                    temperature,
                    max_tokens: maxTokens,
                    stream: true,
                }),
            });

            if (!response.ok) {
                throw new Error(`NVIDIA NIM API error: ${response.status} ${response.statusText}`);
            }

            // Response is already SSE — return it directly (same contract as OpenAI provider)
            return response;
        },
    };
}

module.exports = {
    NvidiaNimProvider,
    createLLM,
    createStreamingLLM,
};
