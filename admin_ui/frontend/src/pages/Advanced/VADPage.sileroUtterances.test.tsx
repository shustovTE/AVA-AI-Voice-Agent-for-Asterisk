// @vitest-environment jsdom

import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import yaml from 'js-yaml';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import VADPage from './VADPage';

const mocks = vi.hoisted(() => ({
    config: { vad: { silero_enabled: true } } as any,
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
vi.mock('../../hooks/useConfirmDialog', () => ({
    useConfirmDialog: () => ({ confirm: vi.fn().mockResolvedValue(true) }),
}));
vi.mock('../../utils/configCache', () => ({
    getCachedConfig: () => ({ config: mocks.config, yamlError: null }),
    loadConfigYaml: vi.fn().mockImplementation(async () => ({ config: mocks.config, yamlError: null })),
}));

const SWITCH = 'Recognizer gets whole utterances cut by Silero';

describe('VADPage Silero utterances', () => {
    beforeEach(() => {
        mocks.post.mockReset();
        mocks.post.mockResolvedValue({ data: { success: true, restart_required: true } });
        mocks.config = { vad: { silero_enabled: true } };
    });

    it('is off by default and then shows the finalize silence', async () => {
        render(<VADPage />);
        expect(await screen.findByLabelText(SWITCH)).not.toBeChecked();
        expect(screen.getByLabelText('STT Finalize Silence (ms)')).toHaveValue(900);
        expect(screen.queryByLabelText('Utterance Pre-roll (ms)')).not.toBeInTheDocument();
    });

    it('switching it on replaces the finalize silence with the utterance fields and saves them', async () => {
        render(<VADPage />);
        fireEvent.click(await screen.findByLabelText(SWITCH));
        expect(screen.queryByLabelText('STT Finalize Silence (ms)')).not.toBeInTheDocument();
        expect(screen.getByLabelText('Utterance Pre-roll (ms)')).toHaveValue(300);
        const max = screen.getByLabelText('Max Utterance (ms)');
        expect(max).toHaveValue(20000);
        fireEvent.change(max, { target: { value: '15000' } });

        fireEvent.click(screen.getByRole('button', { name: /save/i }));

        await waitFor(() => expect(mocks.post).toHaveBeenCalled());
        const call = mocks.post.mock.calls.find(([url]) => url === '/api/config/yaml') as [string, { content: string }];
        const saved = yaml.load(call[1].content) as any;
        expect(saved.vad.silero_stt_utterances).toBe(true);
        expect(saved.vad.silero_utterance_max_ms).toBe(15000);
    });
});
