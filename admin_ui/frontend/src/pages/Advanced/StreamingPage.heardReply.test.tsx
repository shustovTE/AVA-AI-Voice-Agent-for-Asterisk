// @vitest-environment jsdom

import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import yaml from 'js-yaml';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import StreamingPage from './StreamingPage';

const mocks = vi.hoisted(() => ({
    config: { streaming: { pipeline_streaming_overlap: true } } as any,
    post: vi.fn(),
}));

vi.mock('axios', () => ({
    default: { post: mocks.post, get: vi.fn().mockResolvedValue({ data: {} }) },
}));
vi.mock('sonner', () => ({
    toast: { error: vi.fn(), success: vi.fn(), warning: vi.fn(), info: vi.fn() },
}));
vi.mock('../../hooks/useRestartRequired', () => ({
    useRestartRequired: () => ({ restartRequired: false, refetch: vi.fn() }),
}));
vi.mock('../../utils/configCache', () => ({
    getCachedConfig: () => ({ config: mocks.config, yamlError: null }),
    loadConfigYaml: vi.fn().mockImplementation(async () => ({ config: mocks.config, yamlError: null })),
}));

describe('StreamingPage interrupted replies', () => {
    beforeEach(() => {
        mocks.post.mockReset();
        mocks.post.mockResolvedValue({ data: { success: true, restart_required: true } });
        mocks.config = { streaming: { pipeline_streaming_overlap: true } };
    });

    it('shows the heard-reply switch on by default with the default lead', async () => {
        render(<StreamingPage />);
        expect(await screen.findByText('Keep only the heard part of an interrupted reply')).toBeInTheDocument();
        expect(screen.getByLabelText('Heard-audio lead (ms)')).toHaveValue(200);
    });

    it('shows the hangup wait with its default and saves it under streaming.pipeline_hangup_final_wait_ms', async () => {
        render(<StreamingPage />);
        const wait = await screen.findByLabelText('Last words after hangup (ms)');
        expect(wait).toHaveValue(1500);
        fireEvent.change(wait, { target: { value: '2500' } });
        expect(wait).toHaveValue(2500);

        fireEvent.click(screen.getByRole('button', { name: /save/i }));

        await waitFor(() => expect(mocks.post).toHaveBeenCalled());
        const call = mocks.post.mock.calls.find(([url]) => url === '/api/config/yaml') as [string, { content: string }];
        const saved = yaml.load(call[1].content) as any;
        expect(saved.streaming.pipeline_hangup_final_wait_ms).toBe(2500);
    });

    it('saves the lead under streaming.pipeline_heard_reply_lead_ms', async () => {
        render(<StreamingPage />);
        const lead = await screen.findByLabelText('Heard-audio lead (ms)');
        fireEvent.change(lead, { target: { value: '350' } });
        expect(lead).toHaveValue(350);

        fireEvent.click(screen.getByRole('button', { name: /save/i }));

        await waitFor(() => expect(mocks.post).toHaveBeenCalled());
        const call = mocks.post.mock.calls.find(([url]) => url === '/api/config/yaml') as [string, { content: string }];
        const saved = yaml.load(call[1].content) as any;
        expect(saved.streaming.pipeline_heard_reply_lead_ms).toBe(350);
        expect(saved.streaming.pipeline_streaming_overlap).toBe(true);
    });
});

describe('StreamingPage discard of an unheard reply', () => {
    beforeEach(() => {
        mocks.post.mockReset();
        mocks.post.mockResolvedValue({ data: { success: true, restart_required: true } });
        mocks.config = { streaming: { pipeline_streaming_overlap: true } };
    });

    it('is on by default and saves under streaming.pipeline_discard_unheard_reply', async () => {
        render(<StreamingPage />);
        const discard = await screen.findByLabelText('Discard a reply the caller talks over before its first sound');
        expect(discard).toBeChecked();
        fireEvent.click(discard);
        expect(discard).not.toBeChecked();

        fireEvent.click(screen.getByRole('button', { name: /save/i }));

        await waitFor(() => expect(mocks.post).toHaveBeenCalled());
        const call = mocks.post.mock.calls.find(([url]) => url === '/api/config/yaml') as [string, { content: string }];
        const saved = yaml.load(call[1].content) as any;
        expect(saved.streaming.pipeline_discard_unheard_reply).toBe(false);
    });
});

describe('StreamingPage continuation of a reply cut off by nothing', () => {
    beforeEach(() => {
        mocks.post.mockReset();
        mocks.post.mockResolvedValue({ data: { success: true, restart_required: true } });
        mocks.config = { streaming: { pipeline_streaming_overlap: true } };
    });

    it('is on by default and saves the switch and the request text under streaming', async () => {
        render(<StreamingPage />);
        const cont = await screen.findByLabelText('Continue a reply cut off by an unintelligible interruption');
        expect(cont).toBeChecked();
        fireEvent.click(cont);
        expect(cont).not.toBeChecked();
        fireEvent.change(screen.getByLabelText('Continuation request'), {
            target: { value: 'Продолжай с места обрыва.' },
        });

        fireEvent.click(screen.getByRole('button', { name: /save/i }));

        await waitFor(() => expect(mocks.post).toHaveBeenCalled());
        const call = mocks.post.mock.calls.find(([url]) => url === '/api/config/yaml') as [string, { content: string }];
        const saved = yaml.load(call[1].content) as any;
        expect(saved.streaming.pipeline_continue_reply_after_empty_interrupt).toBe(false);
        expect(saved.streaming.pipeline_continue_reply_prompt).toBe('Продолжай с места обрыва.');
    });
});
