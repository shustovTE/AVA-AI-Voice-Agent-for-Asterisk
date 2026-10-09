// @vitest-environment jsdom
import { fireEvent, render, screen } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import axios from 'axios';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';

import EnvPage from './EnvPage';

vi.mock('axios');
vi.mock('../../auth/AuthContext', () => ({
    useAuth: () => ({ token: 'test-token', loading: false }),
}));
vi.mock('../../hooks/useConfirmDialog', () => ({
    useConfirmDialog: () => ({ confirm: vi.fn().mockResolvedValue(false) }),
}));

function mockEnv(env: Record<string, string>) {
    vi.mocked(axios.get).mockImplementation(async url => {
        if (url === '/api/config/env') return { data: env };
        if (url === '/api/config/env/status') return { data: { apply_plan: [], pending_restart: false } };
        if (url === '/api/config/yaml') return { data: { providers: {} } };
        if (typeof url === 'string' && url.startsWith('/api/')) return { data: {} };
        throw new Error(`Unexpected GET ${url}`);
    });
}

const SEGMENTER_LABELS = (prefix: string) => [
    `${prefix} VAD Model Path`,
    `${prefix} VAD Threshold`,
    `${prefix} VAD Min Silence (ms)`,
    `${prefix} VAD Min Speech (ms)`,
    `${prefix} Pre-roll (ms)`,
    `${prefix} Post-roll (ms)`,
    `${prefix} Loudness Target (dBFS)`,
    `${prefix} Max Gain (dB)`,
];

describe('EnvPage offline phrase segmenter settings', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        window.history.replaceState(null, '', '/env#local-ai');
        window.requestAnimationFrame = callback => {
            callback(0);
            return 1;
        };
        Element.prototype.scrollIntoView = vi.fn();
    });

    it('shows the whole segmenter under the onnx-asr backend with the server defaults', async () => {
        mockEnv({ LOCAL_STT_BACKEND: 'onnx_asr', ONNX_ASR_MODEL: 'gigaam-v3-e2e-ctc' });
        render(
            <MemoryRouter>
                <EnvPage />
            </MemoryRouter>
        );

        expect(await screen.findByLabelText('onnx-asr Model')).toHaveValue('gigaam-v3-e2e-ctc');
        expect(screen.getByLabelText('onnx-asr Transducer Decoder')).toHaveValue('cpu');
        expect(screen.getByLabelText('onnx-asr Mel Preprocessor')).toHaveValue('cpu');
        expect(screen.getByLabelText('onnx-asr cuDNN Algorithm Search')).toHaveValue('HEURISTIC');
        expect(screen.getByLabelText('onnx-asr Warm-up')).toHaveValue('true');
        for (const label of SEGMENTER_LABELS('onnx-asr')) {
            expect(screen.getByLabelText(label)).toBeInTheDocument();
        }
        expect(screen.getByLabelText('onnx-asr VAD Threshold')).toHaveValue('0.35');
        expect(screen.getByLabelText('onnx-asr Pre-roll (ms)')).toHaveValue('350');
        expect(screen.getByLabelText('onnx-asr Post-roll (ms)')).toHaveValue('300');
        expect(screen.getByLabelText('onnx-asr Loudness Target (dBFS)')).toHaveValue('-20');
        expect(screen.getByLabelText('onnx-asr Max Gain (dB)')).toHaveValue('24');
        const upsampler = screen.getByLabelText('Local STT 8 kHz Upsampler');
        expect(upsampler).toHaveValue('fir');

        fireEvent.change(upsampler, { target: { value: 'ratecv' } });
        expect(upsampler).toHaveValue('ratecv');
        fireEvent.change(screen.getByLabelText('onnx-asr Loudness Target (dBFS)'), { target: { value: '0' } });
        expect(screen.getByLabelText('onnx-asr Loudness Target (dBFS)')).toHaveValue('0');
        expect(screen.queryByLabelText('Sherpa Post-roll (ms)')).not.toBeInTheDocument();
    });

    it('shows the same segmenter under Sherpa offline and reads saved values', async () => {
        mockEnv({
            LOCAL_STT_BACKEND: 'sherpa',
            SHERPA_MODEL_TYPE: 'offline',
            SHERPA_OFFLINE_POSTROLL_MS: '450',
            SHERPA_OFFLINE_NORMALIZE_DBFS: '-23',
            LOCAL_STT_RESAMPLER: 'ratecv',
        });
        render(
            <MemoryRouter>
                <EnvPage />
            </MemoryRouter>
        );

        expect(await screen.findByLabelText('Sherpa Post-roll (ms)')).toHaveValue('450');
        for (const label of SEGMENTER_LABELS('Sherpa')) {
            expect(screen.getByLabelText(label)).toBeInTheDocument();
        }
        expect(screen.getByLabelText('Sherpa Loudness Target (dBFS)')).toHaveValue('-23');
        expect(screen.getByLabelText('Local STT 8 kHz Upsampler')).toHaveValue('ratecv');
        expect(screen.queryByLabelText('onnx-asr Post-roll (ms)')).not.toBeInTheDocument();
    });

    it('keeps the segmenter out of the online Sherpa settings', async () => {
        mockEnv({ LOCAL_STT_BACKEND: 'sherpa', SHERPA_MODEL_TYPE: 'online' });
        render(
            <MemoryRouter>
                <EnvPage />
            </MemoryRouter>
        );

        expect(await screen.findByLabelText('Sherpa Model Path')).toBeInTheDocument();
        expect(screen.queryByLabelText('Sherpa Post-roll (ms)')).not.toBeInTheDocument();
        expect(screen.queryByLabelText('Local STT 8 kHz Upsampler')).not.toBeInTheDocument();
    });
});
