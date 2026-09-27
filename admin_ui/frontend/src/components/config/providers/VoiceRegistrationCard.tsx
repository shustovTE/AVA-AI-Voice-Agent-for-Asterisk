import React, { useState } from 'react';
import axios from 'axios';
import { Loader2, Mic } from 'lucide-react';
import HelpTooltip from '../../ui/HelpTooltip';
import { FormInput } from '../../ui/FormComponents';

interface VoiceRegistrationCardProps {
    /** Saved YAML key of the provider; undefined while the provider is new (unsaved). */
    providerKey?: string;
    /** Directory inside the ai_engine container the sample is read from. */
    voicesDir: string;
    /** The provider's current `voice`, to offer the registered one in its place. */
    currentVoice?: string;
    onUseVoice: (voice: string) => void;
}

interface RegistrationResult {
    success: boolean;
    message: string;
    voice?: string;
}

/**
 * Register a reference voice on a self-hosted OpenAI-compatible speech server
 * (vLLM-Omni serving Fish Speech S2-Pro and the like). The sample stays on the
 * engine host: the operator names a file in the voices directory and its
 * transcript, the engine uploads both to the server's /audio/voices, and the
 * server answers with the voice name a request can then carry as `voice`.
 */
const VoiceRegistrationCard: React.FC<VoiceRegistrationCardProps> = ({ providerKey, voicesDir, currentVoice, onUseVoice }) => {
    const [file, setFile] = useState('');
    const [name, setName] = useState('');
    const [refText, setRefText] = useState('');
    const [consent, setConsent] = useState('');
    const [busy, setBusy] = useState(false);
    const [result, setResult] = useState<RegistrationResult | null>(null);

    const effectiveName = name.trim() || file.trim().replace(/\.[^.]+$/, '');
    const canSubmit = !!providerKey && !busy && file.trim().length > 0 && refText.trim().length > 0;

    const register = async () => {
        if (!providerKey) return;
        setBusy(true);
        setResult(null);
        try {
            const response = await axios.post(`/api/config/providers/${encodeURIComponent(providerKey)}/voices`, {
                file: file.trim(),
                name: effectiveName || undefined,
                ref_text: refText.trim(),
                consent: consent.trim() || undefined,
            });
            const data = response.data || {};
            setResult({
                success: !!data.success,
                message: data.message || (data.success ? 'Voice registered' : 'Registration failed'),
                voice: data.voice,
            });
        } catch (err: any) {
            setResult({ success: false, message: err?.response?.data?.detail || err?.message || 'Registration failed' });
        } finally {
            setBusy(false);
        }
    };

    return (
        <div className="border border-border rounded-md p-4 space-y-3">
            <div className="flex items-center gap-2">
                <Mic className="w-4 h-4" />
                <span className="font-medium text-sm">Reference voice</span>
                <HelpTooltip content="Registers a sample as a named voice on the speech server, so requests can say voice: <name> instead of carrying the audio. The engine reads the file from its voices directory and uploads it; the server needs 1–30 s of clear speech and the exact transcript. A name that already exists is replaced." />
            </div>
            {!providerKey ? (
                <p className="text-xs text-muted-foreground">Save the provider first; the voice is registered on the saved provider's endpoint.</p>
            ) : (
                <>
                    <p className="text-xs text-muted-foreground">
                        Files are read from <code>{voicesDir}</code> inside the ai_engine container (mount the speech server's voices directory there).
                    </p>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        <FormInput
                            label="Sample file"
                            value={file}
                            onChange={(e) => setFile(e.target.value)}
                            placeholder="anna.wav"
                            tooltip="A bare file name inside the voices directory (wav, mp3, flac, ogg; at most 10 MB)."
                        />
                        <FormInput
                            label="Voice name"
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            placeholder={effectiveName || 'defaults to the file name'}
                            tooltip="The name requests will carry as voice. Defaults to the file name without its extension."
                        />
                    </div>
                    <div className="space-y-1">
                        <label className="text-sm font-medium">Transcript of the sample</label>
                        <textarea
                            className="w-full p-2 text-sm rounded border border-input bg-background min-h-[72px]"
                            value={refText}
                            onChange={(e) => setRefText(e.target.value)}
                            placeholder="Exactly what is said in the sample, with punctuation."
                        />
                        <p className="text-xs text-muted-foreground">Required: the server clones from the sample only together with its exact transcript.</p>
                    </div>
                    <FormInput
                        label="Consent ID (optional)"
                        value={consent}
                        onChange={(e) => setConsent(e.target.value)}
                        placeholder="ava-admin"
                        tooltip="A free-form consent record identifier the server stores with the voice."
                    />
                    <div className="flex items-center gap-3">
                        <button
                            type="button"
                            onClick={register}
                            disabled={!canSubmit}
                            className="inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 border border-input bg-background shadow-sm hover:bg-accent hover:text-accent-foreground h-9 px-4 py-2"
                        >
                            {busy ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Mic className="w-4 h-4 mr-2" />}
                            Register voice
                        </button>
                        {result?.success && result.voice && result.voice !== currentVoice && (
                            <button
                                type="button"
                                onClick={() => onUseVoice(result.voice as string)}
                                className="text-xs text-primary hover:underline"
                            >
                                Use "{result.voice}" as this provider's voice
                            </button>
                        )}
                    </div>
                    {result && (
                        <div className={`p-2 rounded text-xs ${result.success
                            ? 'bg-green-500/10 text-green-600 dark:text-green-400'
                            : 'bg-destructive/10 text-destructive'
                            }`}>
                            {result.message}
                        </div>
                    )}
                </>
            )}
        </div>
    );
};

export default VoiceRegistrationCard;
