import { useState, useEffect } from 'react';
import axios from 'axios';
import { toast } from 'sonner';
import yaml from 'js-yaml';
import { Save, Zap, AlertCircle, RefreshCw, Loader2 } from 'lucide-react';
import { YamlErrorBanner, YamlErrorInfo } from '../../components/ui/YamlErrorBanner';
import { ConfigSection } from '../../components/ui/ConfigSection';
import { ConfigCard } from '../../components/ui/ConfigCard';
import { FormInput, FormSwitch } from '../../components/ui/FormComponents';
import { sanitizeConfigForSave } from '../../utils/configSanitizers';
import { getCachedConfig, loadConfigYaml } from '../../utils/configCache';
import { useRestartRequired } from '../../hooks/useRestartRequired';

const BargeInPage = () => {
    const [config, setConfig] = useState<any>(() => getCachedConfig()?.config ?? {});
    const [loading, setLoading] = useState(() => getCachedConfig() == null);
    const [yamlError, setYamlError] = useState<YamlErrorInfo | null>(() => getCachedConfig()?.yamlError ?? null);
    const [saving, setSaving] = useState(false);
    const { restartRequired, refetch } = useRestartRequired();
    const [restartingEngine, setRestartingEngine] = useState(false);
    const [applyMethod, setApplyMethod] = useState<string>('restart');

    useEffect(() => {
        // Cache-first: seed from the shared cache (no flash on revisit). The write
        // interceptor invalidates the cache on every save, so a background
        // revalidate is unnecessary and could clobber in-progress form edits.
        fetchConfig();
    }, []);

    const fetchConfig = async (force = false) => {
        try {
            const r = await loadConfigYaml(force);
            setConfig(r.config);
            setYamlError(r.yamlError);
        } catch (err) {
            console.error('Failed to load config', err);
            setYamlError(null);
        } finally {
            setLoading(false);
        }
    };

    const handleSave = async () => {
        setSaving(true);
        try {
            const sanitized = sanitizeConfigForSave(config);
            const response = await axios.post('/api/config/yaml', { content: yaml.dump(sanitized) });
            const method = response.data?.recommended_apply_method || 'restart';
            setApplyMethod(method);
            await refetch();
            if (method === 'hot_reload') {
                toast.success('Barge-in configuration saved. Changes can be applied via hot-reload.');
            } else {
                toast.success('Barge-in configuration saved. Restart AI Engine to apply changes.');
            }
        } catch (err) {
            console.error('Failed to save config', err);
            toast.error('Failed to save configuration');
        } finally {
            setSaving(false);
        }
    };

    const handleApplyAIEngine = async (force: boolean = false) => {
        setRestartingEngine(true);
        try {
            // Prefer hot-reload (no dropped calls) when the backend says it suffices (MED-R1).
            if (applyMethod === 'hot_reload') {
                const response = await axios.post('/api/system/containers/ai_engine/reload');

                if (response.data?.restart_required) {
                    setApplyMethod('restart');
                    await refetch();
                    toast.warning('Hot reload applied partially', { description: response.data.message || 'Restart AI Engine to fully apply changes' });
                    return;
                }

                if (response.data?.status === 'success') {
                    await refetch();
                    toast.success('AI Engine hot reloaded! Changes are now active.');
                    return;
                }

                toast.info(`Hot reload response: ${response.data?.message || 'unknown status'}`);
                return;
            }

            const response = await axios.post(`/api/system/containers/ai_engine/restart?force=${force}`);

            if (response.data.status === 'warning') {
                if (!force) {
                    const confirmForce = window.confirm(
                        `${response.data.message}\n\nDo you want to force restart anyway? This may disconnect active calls.`
                    );
                    if (confirmForce) {
                        await handleApplyAIEngine(true);
                    }
                    return;
                }
                toast.warning(response.data.message, { description: 'Force restart is still blocked.' });
                return;
            }

            if (response.data.status === 'degraded') {
                toast.warning('AI Engine restarted but may not be fully healthy', { description: response.data.output || 'Please verify manually' });
                return;
            }

            if (response.data.status === 'success') {
                await refetch();
                toast.success('AI Engine restarted! Changes are now active.');
            }
        } catch (error: any) {
            const actionLabel = applyMethod === 'hot_reload' ? 'hot reload' : 'restart';
            toast.error(`Failed to ${actionLabel} AI Engine`, { description: error.response?.data?.detail || error.message });
        } finally {
            setRestartingEngine(false);
        }
    };

    const updateBargeInConfig = (field: string, value: any) => {
        setConfig({
            ...config,
            barge_in: {
                ...config.barge_in,
                [field]: value
            }
        });
    };

    if (loading) return <div className="p-8 text-center text-muted-foreground">Loading configuration...</div>;

    if (yamlError) return (
        <div className="space-y-6">
            <YamlErrorBanner error={yamlError} />
        </div>
    );

    const bargeInConfig = config.barge_in || {};
    const vadConfig = config.vad || {};
    // Which detector decides pipeline barge-in: Silero VAD (when it owns barge-in),
    // else Asterisk TALK_DETECT, else the engine's own energy detector. The
    // fields of the other detectors change nothing and stay hidden.
    const sileroOwnsBargeIn = Boolean(vadConfig.silero_enabled) && (vadConfig.silero_barge_in ?? true);
    const talkDetectEnabled = bargeInConfig.pipeline_talk_detect_enabled ?? true;
    const energyDetectorActive = !sileroOwnsBargeIn && !talkDetectEnabled;
    const pipelineDetectorLabel = sileroOwnsBargeIn
        ? 'Silero VAD'
        : talkDetectEnabled
            ? 'Asterisk TALK_DETECT'
            : 'the engine energy detector';
    const providerFallbackProviders = Array.isArray(bargeInConfig.provider_fallback_providers)
        ? (bargeInConfig.provider_fallback_providers as string[]).filter(Boolean)
        : [];
    const providerFallbackProvidersStr = providerFallbackProviders.join(', ');
    const providerFallbackMinByProvider =
        bargeInConfig.provider_fallback_min_ms_by_provider &&
        typeof bargeInConfig.provider_fallback_min_ms_by_provider === 'object' &&
        !Array.isArray(bargeInConfig.provider_fallback_min_ms_by_provider)
            ? bargeInConfig.provider_fallback_min_ms_by_provider
            : {};

    const bannerMessage = applyMethod === 'hot_reload'
        ? 'Changes saved. Apply Changes to hot reload AI Engine without dropping active calls.'
        : 'Changes to barge-in configurations require an AI Engine restart to take effect.';

    return (
        <div className="space-y-6">
            {restartRequired && (
                <div className="bg-orange-500/15 border-orange-500/30 border text-yellow-800 dark:text-yellow-500 p-4 rounded-md flex items-center justify-between">
                    <div className="flex items-center">
                        <AlertCircle className="w-5 h-5 mr-2" />
                        {bannerMessage}
                    </div>
                    <button
                        onClick={() => handleApplyAIEngine(false)}
                        disabled={restartingEngine}
                        className="flex items-center text-xs px-3 py-1.5 rounded transition-colors bg-orange-500 text-white hover:bg-orange-600 font-medium disabled:opacity-50"
                    >
                        {restartingEngine ? (
                            <Loader2 className="w-3 h-3 mr-1.5 animate-spin" />
                        ) : (
                            <RefreshCw className="w-3 h-3 mr-1.5" />
                        )}
                        {restartingEngine
                            ? (applyMethod === 'hot_reload' ? 'Applying...' : 'Restarting...')
                            : (applyMethod === 'hot_reload' ? 'Apply Changes' : 'Restart AI Engine')}
                    </button>
                </div>
            )}

            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight">Barge-in Settings</h1>
                    <p className="text-muted-foreground mt-1">
                        Configure how callers can interrupt the AI agent during responses.
                    </p>
                </div>
                <button
                    onClick={handleSave}
                    disabled={saving}
                    className="inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 bg-primary text-primary-foreground shadow hover:bg-primary/90 h-9 px-4 py-2"
                >
                    <Save className="w-4 h-4 mr-2" />
                    {saving ? 'Saving...' : 'Save Changes'}
                </button>
            </div>

            <ConfigSection 
                title="Barge-in Control" 
                description="Allow callers to interrupt the AI while it's speaking."
            >
                <ConfigCard>
                    <div className="space-y-6">
                        <FormSwitch
                            label="Enable Barge-in"
                            description="Allow users to interrupt the AI agent during TTS playback."
                            tooltip="When enabled, the engine immediately flushes/stops local agent audio when it detects caller speech during an agent response."
                            checked={bargeInConfig.enabled ?? true}
                            onChange={(e) => updateBargeInConfig('enabled', e.target.checked)}
                        />

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <FormInput
                                label="Energy Threshold"
                                type="number"
                                value={bargeInConfig.energy_threshold ?? 1000}
                                onChange={(e) => updateBargeInConfig('energy_threshold', parseInt(e.target.value))}
                                tooltip="Caller energy threshold (RMS over PCM16) for provider-owned mode. Higher = less sensitive (fewer false barge-ins), lower = more sensitive (better for quiet callers). For pipelines, see 'Pipeline Energy Threshold' in Advanced settings below."
                            />
                            <FormInput
                                label="Minimum Duration (ms)"
                                type="number"
                                value={bargeInConfig.min_ms ?? 250}
                                onChange={(e) => updateBargeInConfig('min_ms', parseInt(e.target.value))}
                                tooltip="Minimum sustained caller speech time required before triggering barge-in. Higher reduces false triggers but feels less responsive."
                            />
                            <FormInput
                                label="Cooldown (ms)"
                                type="number"
                                value={bargeInConfig.cooldown_ms ?? 500}
                                onChange={(e) => updateBargeInConfig('cooldown_ms', parseInt(e.target.value))}
                                tooltip="Minimum time between barge-in triggers. Prevents repeated triggers from echo/noise after an interruption."
                            />
                            <FormInput
                                label="Post-TTS Protection (ms)"
                                type="number"
                                value={bargeInConfig.post_tts_end_protection_ms ?? 250}
                                onChange={(e) => updateBargeInConfig('post_tts_end_protection_ms', parseInt(e.target.value))}
                                tooltip="Guard window after agent audio ends. Helps avoid self-echo or tail audio being mistaken as caller speech."
                            />
                            <FormInput
                                label="Provider Output Suppress (ms)"
                                type="number"
                                value={bargeInConfig.provider_output_suppress_ms ?? 1200}
                                onChange={(e) => updateBargeInConfig('provider_output_suppress_ms', parseInt(e.target.value))}
                                tooltip="After a barge-in, locally suppress provider audio briefly so previously generated speech doesn’t “resume” mid-sentence."
                            />
                        </div>

                        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
                            <div className="flex items-start">
                                <Zap className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5 mr-3 flex-shrink-0" />
                                <div className="text-sm text-blue-700 dark:text-blue-300">
                                    <p className="font-medium mb-1">Tuning Tips</p>
                                    <ul className="list-disc list-inside space-y-1">
                                        <li><strong>Energy Threshold:</strong> Increase if barge-in is too sensitive (500-1200 typical)</li>
                                        <li><strong>Provider Output Suppress:</strong> Increase if provider resumes speaking pre-barge audio (800-1600ms typical)</li>
                                        <li><strong>Post-TTS Protection:</strong> Increase if you see immediate re-triggers after TTS ends (200-600ms typical)</li>
                                        <li><strong>Talk-Detect / Silero Initial Protection:</strong> Lower (300-800ms) so callers can interrupt a reply sooner; raise it if the agent cuts itself off right after it starts speaking (echo on the line). With Silero VAD, speech that outlasts the window interrupts when it ends, so a long window costs delay, not words. Speech before the reply's first sound never needs it: the reply is discarded instead.</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </ConfigCard>
            </ConfigSection>

            <ConfigSection 
                title="Advanced"
                description="Additional knobs for provider-owned vs pipeline modes."
            >
                <ConfigCard>
                    <details className="space-y-4">
                        <summary className="cursor-pointer text-sm font-medium">Show advanced settings</summary>
                        <div className="space-y-8 pt-4">
                            <div className="space-y-4">
                                <div className="text-sm font-medium">Pipeline protection windows</div>
                                <p className="text-xs text-muted-foreground">
                                    Pipeline barge-in is decided by {pipelineDetectorLabel}. One window applies at a time:
                                    the detector&apos;s own, replaced by the greeting override while the greeting plays when that is longer.
                                </p>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    {energyDetectorActive ? (
                                        <FormInput
                                            label="Initial Protection (ms)"
                                            type="number"
                                            value={bargeInConfig.initial_protection_ms ?? 200}
                                            onChange={(e) => updateBargeInConfig('initial_protection_ms', parseInt(e.target.value))}
                                            tooltip="Engine energy detector (no Silero VAD, TALK_DETECT off): how long after agent output starts caller energy is ignored, against the initial burst and codec artifacts. Silero VAD and TALK_DETECT read the Talk-Detect / Silero window instead."
                                        />
                                    ) : (
                                        <FormInput
                                            label="Talk-Detect / Silero Initial Protection (ms)"
                                            type="number"
                                            value={bargeInConfig.talk_detect_initial_protection_ms ?? 1500}
                                            onChange={(e) => updateBargeInConfig('talk_detect_initial_protection_ms', parseInt(e.target.value))}
                                            tooltip="How long after agent audio starts the caller's speech may not interrupt a reply, against phone echo of the agent's own voice. With Silero VAD the window only defers: speech that starts inside it and is still going when it ends interrupts the reply then, and is recognized whole; speech that stops inside it does not interrupt. Asterisk TALK_DETECT ignores a talking-start inside the window; 0 lets the caller interrupt from the first millisecond (only with echo cancellation on the line). Counted from the stream start, about 200-300 ms before the first audible sound. Speech before the reply's first sound is never blocked by it: the reply is discarded instead (Streaming page → Interrupted Replies)."
                                        />
                                    )}
                                    <FormInput
                                        label="Greeting Protection Override (ms)"
                                        type="number"
                                        value={bargeInConfig.greeting_protection_ms ?? 0}
                                        onChange={(e) => updateBargeInConfig('greeting_protection_ms', parseInt(e.target.value))}
                                        tooltip="Not a third window: while the greeting plays, the active window is replaced by this value when it is longer (it never shortens it). 0 leaves the greeting with the same window as every reply. Useful for short greetings that are cut by early echo."
                                    />
                                </div>
                                <FormSwitch
                                    label="Keep listening while the agent speaks"
                                    description="Pipelines with Silero VAD: the caller's audio keeps reaching the recognizer while the agent speaks, so what they say over a reply is transcribed whether or not it interrupts the reply. The protection window above still decides when speech may interrupt. Off: the recognizer gets silence while the agent is audible."
                                    tooltip="Needs echo cancellation on the line or a phone that does not return the agent's voice, or the agent may transcribe itself. Words spoken into an audible reply are answered after it ends (or after a barge-in cuts it); words spoken before its first sound discard the reply. The utterance that interrupts a reply is recognized whole either way."
                                    checked={bargeInConfig.pipeline_listen_during_playback ?? false}
                                    onChange={(e) => updateBargeInConfig('pipeline_listen_during_playback', e.target.checked)}
                                    disabled={!sileroOwnsBargeIn}
                                />
                            </div>

                            <div className="space-y-4">
                                <div className="text-sm font-medium">Provider-owned mode</div>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <FormInput
                                        label="Provider Initial Protection (ms)"
                                        type="number"
                                        value={bargeInConfig.initial_protection_ms ?? 200}
                                        onChange={(e) => updateBargeInConfig('initial_protection_ms', parseInt(e.target.value))}
                                        tooltip="Full agents (OpenAI Realtime, Deepgram, Google Live, ...): inbound caller audio is dropped for this long after agent output starts, against self-echo (barge_in.initial_protection_ms). Pipelines with Silero VAD or TALK_DETECT do not read it; the engine energy detector does when both are off."
                                    />
                                    <FormSwitch
                                        label="Provider Fallback Enabled"
                                        description="Use local VAD fallback only for providers that don’t emit explicit interruption events."
                                        tooltip="If enabled, the engine can trigger barge-in using local VAD only after media is confirmed and only for the providers listed below."
                                        checked={bargeInConfig.provider_fallback_enabled ?? true}
                                        onChange={(e) => updateBargeInConfig('provider_fallback_enabled', e.target.checked)}
                                    />
                                    <FormInput
                                        label="Provider Fallback Providers"
                                        type="text"
                                        value={providerFallbackProvidersStr}
                                        onChange={(e) =>
                                            updateBargeInConfig(
                                                'provider_fallback_providers',
                                                (e.target.value || '')
                                                    .split(',')
                                                    .map((s) => s.trim())
                                                    .filter(Boolean)
                                            )
                                        }
                                        tooltip="Comma-separated provider names where local fallback may apply (e.g., google_live, deepgram)."
                                    />
                                    <FormInput
                                        label="Grok Fallback Min Duration (ms)"
                                        type="number"
                                        value={providerFallbackMinByProvider.grok ?? 120}
                                        onChange={(e) => {
                                            const value = Number.parseInt(e.target.value, 10);
                                            if (!Number.isFinite(value)) return;
                                            const boundedValue = Math.min(5000, Math.max(40, value));
                                            updateBargeInConfig(
                                                'provider_fallback_min_ms_by_provider',
                                                {
                                                    ...providerFallbackMinByProvider,
                                                    grok: boundedValue,
                                                }
                                            );
                                        }}
                                        min={40}
                                        max={5000}
                                        tooltip="Grok-only local fallback threshold for short commands such as ‘stop’. Other hosted providers continue to use Minimum Duration above. Isolation, VAD, energy, cooldown, and protection guards still apply."
                                    />
                                    <FormInput
                                        label="Suppress Extend (ms)"
                                        type="number"
                                        value={bargeInConfig.provider_output_suppress_extend_ms ?? 600}
                                        onChange={(e) => updateBargeInConfig('provider_output_suppress_extend_ms', parseInt(e.target.value))}
                                        tooltip="While caller keeps speaking after a barge-in, extend suppression so agent doesn’t resume too early."
                                    />
                                    <FormInput
                                        label="Chunk Extend (ms)"
                                        type="number"
                                        value={bargeInConfig.provider_output_suppress_chunk_extend_ms ?? 250}
                                        onChange={(e) => updateBargeInConfig('provider_output_suppress_chunk_extend_ms', parseInt(e.target.value))}
                                        tooltip="While suppressed, extend suppression when provider chunks keep arriving (prevents tail audio from restarting output)."
                                    />
                                </div>
                            </div>

                            <div className="space-y-4">
                                <div className="text-sm font-medium">Pipeline / local_hybrid mode</div>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <FormSwitch
                                        label="Enable TALK_DETECT"
                                        description="Use Asterisk TALK_DETECT for robust barge-in during local file playback."
                                        tooltip="Recommended for local_hybrid: Asterisk DSP detects caller speech even during ARI file playback."
                                        checked={bargeInConfig.pipeline_talk_detect_enabled ?? true}
                                        onChange={(e) => updateBargeInConfig('pipeline_talk_detect_enabled', e.target.checked)}
                                    />
                                    {energyDetectorActive && (
                                        <FormInput
                                            label="Pipeline Min Duration (ms)"
                                            type="number"
                                            value={bargeInConfig.pipeline_min_ms ?? 120}
                                            onChange={(e) => updateBargeInConfig('pipeline_min_ms', parseInt(e.target.value))}
                                            tooltip="Engine energy detector only: minimum sustained caller energy before a pipeline barge-in."
                                        />
                                    )}
                                    {energyDetectorActive && (
                                        <FormInput
                                            label="Pipeline Energy Threshold"
                                            type="number"
                                            value={bargeInConfig.pipeline_energy_threshold ?? 300}
                                            onChange={(e) => updateBargeInConfig('pipeline_energy_threshold', parseInt(e.target.value))}
                                            tooltip="Engine energy detector only: RMS threshold over the caller's PCM16 frames."
                                        />
                                    )}
                                    {talkDetectEnabled && (
                                        <FormInput
                                            label="TALK_DETECT Silence (ms)"
                                            type="number"
                                            value={bargeInConfig.pipeline_talk_detect_silence_ms ?? 1200}
                                            onChange={(e) => updateBargeInConfig('pipeline_talk_detect_silence_ms', parseInt(e.target.value))}
                                            tooltip="Asterisk TALK_DETECT(set) silence threshold in ms. Higher treats more audio as ‘silence’."
                                        />
                                    )}
                                    {talkDetectEnabled && (
                                        <FormInput
                                            label="TALK_DETECT Talking Threshold"
                                            type="number"
                                            value={bargeInConfig.pipeline_talk_detect_talking_threshold ?? 256}
                                            onChange={(e) => updateBargeInConfig('pipeline_talk_detect_talking_threshold', parseInt(e.target.value))}
                                            tooltip="Asterisk TALK_DETECT(set) talking threshold (DSP energy). Higher requires louder speech to trigger."
                                        />
                                    )}
                                </div>
                                {!energyDetectorActive && (
                                    <p className="text-xs text-muted-foreground">
                                        The engine energy detector is not in use ({pipelineDetectorLabel} decides pipeline barge-in), so its
                                        minimum duration, energy threshold and initial protection are not shown.
                                    </p>
                                )}
                            </div>
                        </div>
                    </details>
                </ConfigCard>
            </ConfigSection>

            <ConfigSection 
                title="Current Configuration" 
                description="Summary of your barge-in settings."
            >
                <ConfigCard>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                        <div>
                            <span className="text-muted-foreground">Status:</span>
                            <span className={`ml-2 font-medium ${bargeInConfig.enabled ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                                {bargeInConfig.enabled ? 'Enabled' : 'Disabled'}
                            </span>
                        </div>
                        <div>
                            <span className="text-muted-foreground">Energy Threshold:</span>
                            <span className="ml-2 font-medium">{bargeInConfig.energy_threshold ?? 1000} RMS</span>
                        </div>
                        <div>
                            <span className="text-muted-foreground">Minimum Duration:</span>
                            <span className="ml-2 font-medium">{bargeInConfig.min_ms ?? 250}ms</span>
                        </div>
                        <div>
                            <span className="text-muted-foreground">Post-TTS Protection:</span>
                            <span className="ml-2 font-medium">{bargeInConfig.post_tts_end_protection_ms ?? 250}ms</span>
                        </div>
                    </div>
                </ConfigCard>
            </ConfigSection>
        </div>
    );
};

export default BargeInPage;
