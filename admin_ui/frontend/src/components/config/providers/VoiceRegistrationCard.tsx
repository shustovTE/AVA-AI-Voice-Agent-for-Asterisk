import React, { useRef, useState } from 'react';
import axios from 'axios';
import { List, Loader2, Mic } from 'lucide-react';
import HelpTooltip from '../../ui/HelpTooltip';
import { FormInput } from '../../ui/FormComponents';

interface VoiceRegistrationCardProps {
    /** Saved YAML key of the provider; undefined while the provider is new (unsaved). */
    providerKey?: string;
    /** The provider's current `voice`, to offer a registered one in its place. */
    currentVoice?: string;
    onUseVoice: (voice: string) => void;
}

interface RegistrationResult {
    success: boolean;
    message: string;
    voice?: string;
}

interface RegisteredVoice {
    name: string;
    created_at?: number | string | null;
    file_size?: number | null;
    ref_text?: string | null;
    consent?: string | null;
}

interface VoiceList {
    voices: RegisteredVoice[];
    builtin: string[];
    message: string;
}

const formatCreated = (value: RegisteredVoice['created_at']): string => {
    const raw = Number(value);
    if (!value || Number.isNaN(raw) || raw <= 0) return '';
    const ms = raw > 1e12 ? raw : raw * 1000;
    return new Date(ms).toLocaleString();
};

const formatSize = (value: RegisteredVoice['file_size']): string => {
    const raw = Number(value);
    if (!value || Number.isNaN(raw) || raw <= 0) return '';
    return `${Math.round(raw / 1024)} KB`;
};

/**
 * Register a reference voice on a self-hosted OpenAI-compatible speech server
 * (vLLM-Omni serving Fish Speech S2-Pro and the like), and list what it holds.
 * The sample is picked in the browser; the Admin UI relays it to the engine,
 * which shares a host with the server and uploads it to /audio/voices. The
 * server answers with the voice name a request can then carry as `voice`.
 */
const VoiceRegistrationCard: React.FC<VoiceRegistrationCardProps> = ({ providerKey, currentVoice, onUseVoice }) => {
    const fileInput = useRef<HTMLInputElement | null>(null);
    const [file, setFile] = useState<File | null>(null);
    const [name, setName] = useState('');
    const [refText, setRefText] = useState('');
    const [consent, setConsent] = useState('');
    const [busy, setBusy] = useState(false);
    const [result, setResult] = useState<RegistrationResult | null>(null);
    const [listing, setListing] = useState(false);
    const [list, setList] = useState<VoiceList | null>(null);
    const [listError, setListError] = useState<string | null>(null);

    const effectiveName = name.trim() || (file ? file.name.replace(/\.[^.]+$/, '') : '');
    const canSubmit = !!providerKey && !busy && !!file && refText.trim().length > 0;

    const register = async () => {
        if (!providerKey || !file) return;
        setBusy(true);
        setResult(null);
        try {
            const form = new FormData();
            form.append('audio_sample', file, file.name);
            form.append('ref_text', refText.trim());
            if (effectiveName) form.append('voice_name', effectiveName);
            if (consent.trim()) form.append('consent', consent.trim());
            const response = await axios.post(`/api/config/providers/${encodeURIComponent(providerKey)}/voices`, form);
            const data = response.data || {};
            setResult({
                success: !!data.success,
                message: data.message || (data.success ? 'Voice registered' : 'Registration failed'),
                voice: data.voice,
            });
            if (data.success && list) {
                await loadList();
            }
        } catch (err: any) {
            setResult({ success: false, message: err?.response?.data?.detail || err?.message || 'Registration failed' });
        } finally {
            setBusy(false);
        }
    };

    const loadList = async () => {
        if (!providerKey) return;
        setListing(true);
        setListError(null);
        try {
            const response = await axios.get(`/api/config/providers/${encodeURIComponent(providerKey)}/voices`);
            const data = response.data || {};
            if (!data.success) {
                setList(null);
                setListError(data.message || 'The server did not list its voices');
            } else {
                setList({
                    voices: Array.isArray(data.voices) ? data.voices : [],
                    builtin: Array.isArray(data.builtin) ? data.builtin : [],
                    message: data.message || '',
                });
            }
        } catch (err: any) {
            setList(null);
            setListError(err?.response?.data?.detail || err?.message || 'Could not list voices');
        } finally {
            setListing(false);
        }
    };

    return (
        <div className="border border-border rounded-md p-4 space-y-3">
            <div className="flex items-center gap-2">
                <Mic className="w-4 h-4" />
                <span className="font-medium text-sm">Reference voice</span>
                <HelpTooltip content="Registers a sample as a named voice on the speech server, so requests can say voice: <name> instead of carrying the audio. The file is uploaded from this browser through the engine; the server needs 1–30 s of clear speech (at most 10 MB) and the exact transcript. A name that already exists is replaced." />
            </div>
            {!providerKey ? (
                <p className="text-xs text-muted-foreground">Save the provider first; voices are registered on the saved provider's endpoint.</p>
            ) : (
                <>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        <div className="space-y-1">
                            <label className="text-sm font-medium">Sample file</label>
                            <input
                                ref={fileInput}
                                type="file"
                                accept=".wav,.mp3,.flac,.ogg,.aac,.webm,.mp4,.m4a,audio/*"
                                className="block w-full text-sm text-muted-foreground file:mr-3 file:rounded-md file:border file:border-input file:bg-background file:px-3 file:py-1.5 file:text-sm file:font-medium hover:file:bg-accent"
                                onChange={(e) => setFile(e.target.files && e.target.files[0] ? e.target.files[0] : null)}
                            />
                            <p className="text-xs text-muted-foreground">wav, mp3, flac, ogg, aac, webm or m4a; 1–30 s of one speaker, at most 10 MB.</p>
                        </div>
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
                    <div className="flex flex-wrap items-center gap-3">
                        <button
                            type="button"
                            onClick={register}
                            disabled={!canSubmit}
                            className="inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 border border-input bg-background shadow-sm hover:bg-accent hover:text-accent-foreground h-9 px-4 py-2"
                        >
                            {busy ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Mic className="w-4 h-4 mr-2" />}
                            Register voice
                        </button>
                        <button
                            type="button"
                            onClick={loadList}
                            disabled={listing}
                            className="inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 border border-input bg-background shadow-sm hover:bg-accent hover:text-accent-foreground h-9 px-4 py-2"
                        >
                            {listing ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <List className="w-4 h-4 mr-2" />}
                            Registered voices
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
                    {listError && (
                        <div className="p-2 rounded text-xs bg-destructive/10 text-destructive">{listError}</div>
                    )}
                    {list && (
                        <div className="space-y-2">
                            <p className="text-xs text-muted-foreground">{list.message}</p>
                            {list.voices.length === 0 ? (
                                <p className="text-sm text-muted-foreground italic">No registered voices on this server yet.</p>
                            ) : (
                                <table className="w-full text-xs">
                                    <thead>
                                        <tr className="text-left text-muted-foreground">
                                            <th className="py-1 pr-2 font-medium">Voice</th>
                                            <th className="py-1 pr-2 font-medium">Registered</th>
                                            <th className="py-1 pr-2 font-medium">Sample</th>
                                            <th className="py-1 pr-2 font-medium">Transcript</th>
                                            <th className="py-1 font-medium"></th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {list.voices.map((voice) => (
                                            <tr key={voice.name} className="border-t border-border align-top">
                                                <td className="py-1 pr-2 font-mono">
                                                    {voice.name}
                                                    {voice.name === currentVoice && <span className="ml-1 text-muted-foreground">(current)</span>}
                                                </td>
                                                <td className="py-1 pr-2 whitespace-nowrap">{formatCreated(voice.created_at)}</td>
                                                <td className="py-1 pr-2 whitespace-nowrap">{formatSize(voice.file_size)}</td>
                                                <td className="py-1 pr-2 text-muted-foreground" title={voice.ref_text || ''}>
                                                    {(voice.ref_text || '').length > 80 ? `${(voice.ref_text || '').slice(0, 80)}…` : (voice.ref_text || '')}
                                                </td>
                                                <td className="py-1 text-right">
                                                    {voice.name !== currentVoice && (
                                                        <button type="button" onClick={() => onUseVoice(voice.name)} className="text-primary hover:underline">
                                                            Use
                                                        </button>
                                                    )}
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            )}
                        </div>
                    )}
                </>
            )}
        </div>
    );
};

export default VoiceRegistrationCard;
