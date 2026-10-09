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

const SELECT = 'Scoring Rate';

async function saved(): Promise<any> {
    fireEvent.click(screen.getByRole('button', { name: /save/i }));
    await waitFor(() => expect(mocks.post).toHaveBeenCalled());
    const call = mocks.post.mock.calls.find(([url]) => url === '/api/config/yaml') as [string, { content: string }];
    return yaml.load(call[1].content);
}

describe('VADPage Silero scoring rate', () => {
    beforeEach(() => {
        mocks.post.mockReset();
        mocks.post.mockResolvedValue({ data: { success: true, restart_required: true } });
        mocks.config = { vad: { silero_enabled: true } };
    });

    it("defaults to the line's own rate", async () => {
        render(<VADPage />);
        expect(await screen.findByLabelText(SELECT)).toHaveValue('');
    });

    it('saves 16 kHz as silero_sample_rate: 16000', async () => {
        render(<VADPage />);
        fireEvent.change(await screen.findByLabelText(SELECT), { target: { value: '16000' } });
        expect((await saved()).vad.silero_sample_rate).toBe(16000);
    });

    it("going back to the line's own rate removes the key", async () => {
        mocks.config = { vad: { silero_enabled: true, silero_sample_rate: 16000 } };
        render(<VADPage />);
        const select = await screen.findByLabelText(SELECT);
        expect(select).toHaveValue('16000');
        fireEvent.change(select, { target: { value: '' } });
        expect((await saved()).vad).not.toHaveProperty('silero_sample_rate');
    });
});
