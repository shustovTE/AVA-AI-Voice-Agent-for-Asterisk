// @vitest-environment jsdom

import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import axios from 'axios';
import yaml from 'js-yaml';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import ProvidersPage from './ProvidersPage';

const mocks = vi.hoisted(() => ({
    config: {} as Record<string, unknown>,
    refetch: vi.fn().mockResolvedValue(undefined),
    confirm: vi.fn().mockResolvedValue(true),
    loadConfigYaml: vi.fn(),
    toastError: vi.fn(),
}));

vi.mock('axios');
vi.mock('sonner', () => ({
    toast: {
        error: mocks.toastError,
        success: vi.fn(),
        warning: vi.fn(),
        info: vi.fn(),
    },
}));
vi.mock('../hooks/useConfirmDialog', () => ({
    useConfirmDialog: () => ({ confirm: mocks.confirm }),
}));
vi.mock('../hooks/useRestartRequired', () => ({
    useRestartRequired: () => ({
        restartRequired: false,
        refetch: mocks.refetch,
    }),
}));
vi.mock('../utils/configCache', () => ({
    getCachedConfig: () => ({ config: mocks.config, yamlError: null }),
    loadConfigYaml: mocks.loadConfigYaml,
}));

/**
 * A save merges the form over the stored provider so settings the form cannot
 * represent survive. That merge used to resurrect any field the operator had
 * just deleted, and for an OpenAI-compatible provider every unknown key is
 * forwarded to the endpoint, where a stale one can fail the request.
 */
describe('ProvidersPage field removal', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        mocks.confirm.mockResolvedValue(true);
        mocks.loadConfigYaml.mockImplementation(async () => ({
            config: mocks.config,
            yamlError: null,
        }));
        vi.mocked(axios.get).mockResolvedValue({ data: {} });
        vi.mocked(axios.post).mockResolvedValue({ data: {}, status: 200 });
        mocks.config = {
            providers: {
                native_llm: {
                    type: 'openai',
                    capabilities: ['llm'],
                    enabled: true,
                    chat_base_url: 'https://api.mistral.ai/v1',
                    chat_model: 'mistral-small-latest',
                    CACHE_KEY: 'prompt-1',
                    prompt_cache_key: 'prompt-1',
                },
            },
            default_provider: 'native_llm',
        };
    });

    const savedProvider = () => {
        const saveCall = vi.mocked(axios.post).mock.calls.find(([url]) => url === '/api/config/yaml');
        expect(saveCall).toBeDefined();
        const body = saveCall?.[1] as { content: string };
        const saved = yaml.load(body.content) as {
            providers: Record<string, Record<string, unknown>>;
        };
        return saved.providers.native_llm;
    };

    it('drops a deleted field instead of restoring it from the saved provider', async () => {
        render(
            <MemoryRouter>
                <ProvidersPage />
            </MemoryRouter>,
        );

        fireEvent.click(await screen.findByTitle('Settings'));
        const dialog = await screen.findByRole('dialog', { name: 'Edit Provider: native_llm' });

        const staleKey = within(dialog).getByDisplayValue('CACHE_KEY');
        // The key input sits in its own column; the remove button is a sibling
        // of that column, one level up.
        const row = staleKey.closest('div')?.parentElement;
        expect(row).not.toBeNull();
        fireEvent.click(within(row as HTMLElement).getByRole('button'));

        fireEvent.click(within(dialog).getByRole('button', { name: 'Save Changes' }));

        await waitFor(() => {
            expect(axios.post).toHaveBeenCalledWith(
                '/api/config/yaml',
                expect.objectContaining({ content: expect.any(String) }),
            );
        });

        const provider = savedProvider();
        expect(provider).not.toHaveProperty('CACHE_KEY');
        // Untouched fields, including the one that looks similar, stay put.
        expect(provider.prompt_cache_key).toBe('prompt-1');
        expect(provider.chat_model).toBe('mistral-small-latest');
    });

    it('keeps every field when nothing is deleted', async () => {
        render(
            <MemoryRouter>
                <ProvidersPage />
            </MemoryRouter>,
        );

        fireEvent.click(await screen.findByTitle('Settings'));
        const dialog = await screen.findByRole('dialog', { name: 'Edit Provider: native_llm' });
        fireEvent.click(within(dialog).getByRole('button', { name: 'Save Changes' }));

        await waitFor(() => {
            expect(axios.post).toHaveBeenCalledWith(
                '/api/config/yaml',
                expect.objectContaining({ content: expect.any(String) }),
            );
        });

        const provider = savedProvider();
        expect(provider.CACHE_KEY).toBe('prompt-1');
        expect(provider.prompt_cache_key).toBe('prompt-1');
    });
});
