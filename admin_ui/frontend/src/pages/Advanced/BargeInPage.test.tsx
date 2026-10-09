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

describe('BargeInPage shows only the fields of the detector that decides pipeline barge-in', () => {
    beforeEach(() => {
        mocks.post.mockReset();
        mocks.post.mockResolvedValue({ data: { success: true, restart_required: true } });
    });

    it('with Silero VAD owning barge-in hides the energy detector and offers listening during playback', async () => {
        mocks.config = {
            barge_in: { enabled: true, pipeline_talk_detect_enabled: false },
            vad: { silero_enabled: true, silero_barge_in: true },
        };
        render(<BargeInPage />);
        expect(await screen.findByLabelText(FIELD)).toHaveValue(1500);
        expect(screen.queryByLabelText('Initial Protection (ms)')).not.toBeInTheDocument();
        expect(screen.queryByLabelText('Pipeline Energy Threshold')).not.toBeInTheDocument();
        expect(screen.queryByLabelText('Pipeline Min Duration (ms)')).not.toBeInTheDocument();
        expect(screen.queryByLabelText('TALK_DETECT Silence (ms)')).not.toBeInTheDocument();
        expect(screen.getByLabelText('Greeting Protection Override (ms)')).toHaveValue(0);
        // Full agents keep their own window, under their own heading.
        expect(screen.getByLabelText('Provider Initial Protection (ms)')).toHaveValue(200);

        const listen = screen.getByLabelText('Keep listening while the agent speaks');
        expect(listen).not.toBeChecked();
        fireEvent.click(listen);
        fireEvent.click(screen.getByRole('button', { name: /save/i }));

        await waitFor(() => expect(mocks.post).toHaveBeenCalled());
        const [, body] = mocks.post.mock.calls.find(([u]) => u === '/api/config/yaml') as [string, { content: string }];
        const saved = yaml.load(body.content) as any;
        expect(saved.barge_in.pipeline_listen_during_playback).toBe(true);
        expect(saved.barge_in.initial_protection_ms).toBeUndefined();
    });

    it('without Silero VAD and TALK_DETECT shows the energy detector fields instead', async () => {
        mocks.config = {
            barge_in: { enabled: true, pipeline_talk_detect_enabled: false, initial_protection_ms: 250 },
            vad: { silero_enabled: false },
        };
        render(<BargeInPage />);
        expect(await screen.findByLabelText('Initial Protection (ms)')).toHaveValue(250);
        expect(screen.getByLabelText('Pipeline Energy Threshold')).toHaveValue(300);
        expect(screen.queryByLabelText(FIELD)).not.toBeInTheDocument();
        expect(screen.getByLabelText('Keep listening while the agent speaks')).toBeDisabled();
    });
});
