import { describe, expect, it } from 'vitest';
import { MODULAR_SUBTYPES, inferSubtype } from './modularProviderSubtypes';

describe('modular LLM provider subtypes', () => {
    it('offers a first-class DeepSeek preset with current official defaults', () => {
        const deepseek = MODULAR_SUBTYPES.llm.find(subtype => subtype.id === 'deepseek');

        expect(deepseek).toBeDefined();
        expect(deepseek?.yamlType).toBe('openai');
        expect(deepseek?.fields).toEqual(
            expect.arrayContaining([
                expect.objectContaining({
                    key: 'chat_base_url',
                    default: 'https://api.deepseek.com',
                }),
                expect.objectContaining({
                    key: 'chat_model',
                    default: 'deepseek-v4-flash',
                }),
            ])
        );
    });

    it('recognizes an existing DeepSeek OpenAI-compatible configuration', () => {
        expect(
            inferSubtype({
                type: 'openai',
                capabilities: ['llm'],
                chat_base_url: 'https://api.deepseek.com',
                chat_model: 'deepseek-v4-pro',
            })?.id
        ).toBe('deepseek');
    });
});

describe('modular TTS provider subtypes', () => {
    it('exposes the ElevenLabs proxy settings so they are reachable outside the full-agent form', () => {
        const elevenlabs = MODULAR_SUBTYPES.tts.find(subtype => subtype.id === 'elevenlabs');

        expect(elevenlabs).toBeDefined();
        expect(elevenlabs?.yamlType).toBe('elevenlabs');
        expect(elevenlabs?.fields).toEqual(
            expect.arrayContaining([
                expect.objectContaining({ key: 'proxy', type: 'text', required: false }),
                expect.objectContaining({ key: 'keepalive_timeout_sec', type: 'number', required: false }),
            ])
        );
    });

    it('leaves both routing fields empty by default, which means a direct connection', () => {
        const elevenlabs = MODULAR_SUBTYPES.tts.find(subtype => subtype.id === 'elevenlabs');
        const routing = elevenlabs?.fields.filter(
            field => field.key === 'proxy' || field.key === 'keepalive_timeout_sec'
        );

        expect(routing).toHaveLength(2);
        routing?.forEach(field => expect(field.default).toBeUndefined());
    });
});
