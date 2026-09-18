// @vitest-environment jsdom

import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import yaml from 'js-yaml';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import BargeInPage from './BargeInPage';

const mocks = vi.hoisted(() => ({
    config: {
        barge_in: {
            enabled: true,
            initial_protection_ms: 200,
            greeting_protection_ms: 0,
        },
    } as any,
    post: vi.fn(),
}));

vi.mock('axios', () => ({
    default: { post: mocks.post, get: vi.fn() },
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

const FIELD = 'Talk-Detect / Silero Initial Protection (ms)';

describe('BargeInPage talk-detect / Silero initial protection', () => {
    beforeEach(() => {
        mocks.post.mockReset();
        mocks.post.mockResolvedValue({ data: { success: true, restart_required: true } });
        mocks.config = { barge_in: { enabled: true, initial_protection_ms: 200, greeting_protection_ms: 0 } };
    });

    it('shows the engine default when the key is absent and keeps an explicit 0', async () => {
        const { unmount } = render(<BargeInPage />);
        expect(await screen.findByLabelText(FIELD)).toHaveValue(1500);
        unmount();

        mocks.config = { barge_in: { enabled: true, talk_detect_initial_protection_ms: 0 } };
        render(<BargeInPage />);
        expect(await screen.findByLabelText(FIELD)).toHaveValue(0);
    });

    it('saves the value under barge_in.talk_detect_initial_protection_ms', async () => {
        render(<BargeInPage />);
        const field = await screen.findByLabelText(FIELD);
        fireEvent.change(field, { target: { value: '400' } });
        expect(field).toHaveValue(400);

        fireEvent.click(screen.getByRole('button', { name: /save/i }));

        await waitFor(() => expect(mocks.post).toHaveBeenCalled());
        const [url, body] = mocks.post.mock.calls.find(([u]) => u === '/api/config/yaml') as [string, { content: string }];
        expect(url).toBe('/api/config/yaml');
        const saved = yaml.load(body.content) as any;
        expect(saved.barge_in.talk_detect_initial_protection_ms).toBe(400);
        expect(saved.barge_in.initial_protection_ms).toBe(200);
    });
});
