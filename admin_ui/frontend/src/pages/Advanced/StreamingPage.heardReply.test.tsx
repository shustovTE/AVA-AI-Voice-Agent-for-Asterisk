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
