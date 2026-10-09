# Security Policy

## Supported Versions

**Support policy**: we support the **two most recent minor release trains**. When a new minor ships, the oldest supported train reaches end of support. This is an honest reflection of what a small maintainer team can patch — older versions still work, but security fixes land only on supported trains.

| Version | Supported          | End of Support                      |
| ------- | ------------------ | ----------------------------------- |
| 7.5.x   | :white_check_mark: | Current                             |
| 7.4.x   | :white_check_mark: | When the second following minor ships |
| 7.3.x   | :x:                | Ended 2026-07-22                    |
| 7.2.x   | :x:                | Ended 2026-07-16                    |
| 7.1.x   | :x:                | Ended 2026-07-02                    |
| 7.0.x   | :x:                | Ended 2026-07-02                    |
| 6.x     | :x:                | Ended 2026-07-02                    |
| < 6.0   | :x:                | Ended                               |

**Recommendation**: Always upgrade to the latest release for the most recent security patches and features. Upgrade guides: [docs/MIGRATION.md](docs/MIGRATION.md) and [docs/OPERATOR_MIGRATION.md](docs/OPERATOR_MIGRATION.md).

---

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please follow our responsible disclosure process.

### Reporting Process

1. **DO NOT** open a public GitHub issue for security vulnerabilities
2. **Email** security reports to: support@jugaar.llc
3. **Include** the following in your report:
   - Description of the vulnerability
   - Steps to reproduce the issue
   - Affected versions
   - Potential impact
   - Any suggested fixes (optional)

### What to Expect

| Timeline | Action |
|----------|--------|
| **Within 48 hours** | We will acknowledge receipt of your report |
| **Within 7 days** | We will provide an initial assessment and timeline |
| **Within 30 days** | We will aim to release a patch or mitigation guidance |

### Response SLAs

- **Critical vulnerabilities** (remote code execution, authentication bypass):
  - Acknowledgment: 24 hours
  - Patch release: 7-14 days
  
- **High vulnerabilities** (privilege escalation, data exposure):
  - Acknowledgment: 48 hours
  - Patch release: 14-30 days
  
- **Medium/Low vulnerabilities**:
  - Acknowledgment: 72 hours
  - Patch release: Next scheduled release

---

## Security Best Practices

### 1. Credentials Management

**CRITICAL**: Never commit credentials to version control.

- Store all secrets in `.env` file (gitignored by default)
- Required secrets:
  ```bash
  ASTERISK_ARI_USERNAME=your_username
  ASTERISK_ARI_PASSWORD=your_secure_password
  OPENAI_API_KEY=sk-...
  DEEPGRAM_API_KEY=...
  ```
- Use strong, unique passwords (minimum 16 characters)
- Rotate API keys every 90 days
- Never include `.env` in Docker images

### 2. Network Security

**Default Configuration** (Secure - all services bind to localhost):
| Service | Default Bind | Port | Remote Opt-in |
|---------|-------------|------|---------------|
| ai-engine Health | `127.0.0.1` (code) / `0.0.0.0` (shipped compose — see note) | 15000 | `HEALTH_BIND_HOST=0.0.0.0` |
| Admin UI | `127.0.0.1` | 3003 | `UVICORN_HOST=0.0.0.0` |
| Local AI Server | `127.0.0.1` | 8765 | `LOCAL_WS_HOST=0.0.0.0` |
| RTP Server | `127.0.0.1` | 18080 | `EXTERNAL_MEDIA_RTP_HOST=0.0.0.0` |
| AudioSocket | `127.0.0.1` | 8090 | `AUDIOSOCKET_HOST=0.0.0.0` |

> **Shipped docker-compose note:** `docker-compose.yml` sets `HEALTH_BIND_HOST=0.0.0.0`
> for the ai-engine health server (port 15000) so the Admin UI container can reach
> `/sessions/stats` and `/reload`. On host-network deployments this exposes port 15000
> on all host interfaces. The health server carries no secrets but does expose
> operational state and a `/reload` trigger — firewall port 15000 from untrusted
> networks, or, if you do not use the Admin UI, set `HEALTH_BIND_HOST=127.0.0.1`
> in the `ai_engine` service's `environment:` block in `docker-compose.yml`.
> (Setting it only in `.env` has no effect: the compose `environment:` value takes
> precedence over `env_file` — see the Docker Compose env-var precedence docs.)

**If Remote Access Required**:
```bash
# .env file - explicit opt-in required
HEALTH_BIND_HOST=0.0.0.0         # ai-engine health endpoints
HEALTH_API_TOKEN=<strong-token>  # Required for /reload, /mcp/test/* from remote
UVICORN_HOST=0.0.0.0             # Admin UI (REQUIRES JWT_SECRET!)
JWT_SECRET=<openssl rand -hex 32>  # Required when UVICORN_HOST != localhost
LOCAL_WS_HOST=0.0.0.0            # Local AI Server
LOCAL_WS_AUTH_TOKEN=<token>      # Required when LOCAL_WS_HOST != localhost
```

**Firewall Rules** (if binding to 0.0.0.0):
```bash
# Only allow from trusted IPs
sudo ufw allow from 10.0.1.5 to any port 18080  # RTP
sudo ufw allow from 10.0.1.5 to any port 8090   # AudioSocket
sudo ufw allow from 10.0.1.5 to any port 15000  # Health (if exposed)
```

### 2.1 Admin UI Security

**⚠️ CRITICAL: Admin UI is a Control Plane**

The Admin UI has Docker socket access for container management. If exposed remotely without proper security, this is effectively **root-equivalent access** to the host.

**Risk Summary**:
| Exposure | Risk Level | Impact |
|----------|------------|--------|
| Localhost only | Low | Expected use case |
| LAN without auth | **Critical** | Full host compromise possible |
| Internet without auth | **Critical** | Immediate compromise |
| Internet with JWT only | High | Brute-force/leak risk |
| Reverse proxy + mTLS | Low | Recommended production setup |

**Security Requirements**:
1. **Never expose directly to internet** - Always use reverse proxy with authentication
2. **JWT_SECRET is mandatory** - Service refuses to start if binding non-localhost without JWT_SECRET
3. **Network isolation** - Place admin-ui on management network only
4. **Least privilege** - Consider read-only Docker socket mounts if container management not needed
5. **Rotate the admin password** - AVA generates a one-time admin password on first start and forces a change at first login; never run with a shared or unrotated admin credential.

**Docker Socket Hardening**:
```yaml
# docker-compose.yml - Read-only socket (if only monitoring needed)
services:
  admin-ui:
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro  # Read-only
```

```bash
# Alternative: Use docker-socket-proxy for granular control
# https://github.com/Tecnativa/docker-socket-proxy
docker run -d --name docker-proxy \
  -v /var/run/docker.sock:/var/run/docker.sock:ro \
  -e CONTAINERS=1 -e INFO=1 -e IMAGES=0 -e EXEC=0 \
  tecnativa/docker-socket-proxy
```

**Recommended Production Setup**:
```nginx
# nginx reverse proxy with client cert auth (mTLS)
server {
    listen 443 ssl;
    ssl_certificate /etc/nginx/server.crt;
    ssl_certificate_key /etc/nginx/server.key;
    ssl_client_certificate /etc/nginx/client-ca.crt;
    ssl_verify_client on;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=admin:10m rate=10r/s;
    limit_req zone=admin burst=20 nodelay;
    
    location / {
        proxy_pass http://127.0.0.1:3003;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

**Audit Logging** (recommended):
```bash
# Enable Docker daemon audit logging
# /etc/docker/daemon.json
{
  "log-driver": "json-file",
  "log-opts": {"max-size": "10m", "max-file": "3"}
}
```

### 2.2 Sensitive Endpoint Protection

The following ai-engine endpoints require authorization:
- `POST /reload` - Hot-reload configuration
- `POST /mcp/test/{server_id}` - Test MCP server connections

**Authorization Methods**:
1. **Localhost access** - Automatically authorized from 127.0.0.1
2. **API Token** - Set `HEALTH_API_TOKEN` and include `Authorization: Bearer <token>` header

```bash
# Remote reload with token
curl -X POST http://ai_engine:15000/reload \
  -H "Authorization: Bearer $HEALTH_API_TOKEN"
```

### 3. Docker Security

**Run as Non-Root**:
- Containers run as `appuser` (non-root) by default
- Never override this with `user: root`

**Keep Base Images Updated**:
```bash
# Check for updates
docker pull python:3.11@sha256:e8ab764baee5109566456913b42d7d4ad97c13385e4002973c896e1dd5f01146

# Rebuild
docker compose build --no-cache
```

#### Public detector model files

`src/core/model_fetch.py`, `scripts/fetch_silero_vad.sh`, and
`scripts/fetch_smart_turn.sh` intentionally publish downloaded model files as
`0644` (`rw-r--r--`). This policy applies to the public Silero VAD and Smart Turn
ONNX weights, not to credentials, recordings, transcripts, or private models.
Reading these weights discloses no application secrets or call data; group and
other users receive no file write or execute permission.

The engine runs as non-root `appuser` (`Dockerfile`), and the Compose template
bind-mounts `./models` at `/app/models`. A host-side downloader may have a
different numeric UID and group membership. Replacing `0644` with `0600` would
prevent that container user from reading a host-owned model; `0640` would also
require a reliably shared GID. `0644` preserves the documented host-preseed
workflow without changing ownership or running the engine as root. Directory
traversal permissions and any ACL/SELinux policy must still allow the read.

The automatic downloads use fixed public HTTPS URLs and SHA-256 values in
`src/core/silero_vad.py` and `src/core/smart_turn.py`. Silero is version-tagged;
Smart Turn's URL uses upstream `main`, but its expected digest is fixed. An
upstream content change therefore fails verification. Both the Python helper
and the shell scripts start with a private `mkstemp`/`mktemp` file next to the
destination, verify the digest before setting `0644`, and rename it into place.
A failed transfer or digest check leaves an existing destination intact and
removes the temporary file.

**Trust boundary:** file read permission is not permission to replace the model.
Only trusted deployment users/services may write model files or their parent
directories (including ancestors), whether through ownership, groups, ACLs, or
another container mount. The installer/preflight use shared directories with
mode `2775`; membership of that writable group is consequently trusted. The
Admin UI's model-permission repair can also grant group write access to existing
files. Do not add untrusted users to those groups or make the tree world-writable.
An attacker with directory write access can replace a file regardless of whether
the file is `0644` or `0600`; atomic rename and a download checksum do not defend
against such an attacker. In particular, `ensure_file()` deliberately accepts a
nonempty operator-provided model with a different digest, logging a warning.
It is not strict integrity enforcement for files already on disk.

For deployments that pre-provision every required model, a read-only engine
mount (`./models:/app/models:ro`) is an optional additional boundary. Disable
`vad.silero_auto_download` and `vad.smart_turn_auto_download` and provision both
files first. This prevents engine-side downloads/updates and does not prevent
trusted host processes or other writable mounts from changing the same files.
Private models need a separate ownership/shared-group/ACL policy; the public
download helper must not be reused for confidential assets. Existing manually
provisioned model permissions are not changed by `ensure_file()`.

**CodeQL triage:** `py/overly-permissive-file` checks the permission mask without
classifying the file's contents. The world-readable observation is correct, but
for these public weights the security finding is a false positive. Keep the
literal `0644` and retain analysis of this file and rule. Do not use a broad
`paths-ignore` entry, disable the query, filter its SARIF results, or obscure the
mode to hide the alert.

For [alert #11 in shustovTE/AVA-AI-Voice-Agent-for-Asterisk](https://github.com/shustovTE/AVA-AI-Voice-Agent-for-Asterisk/security/code-scanning/11),
reported on [PR #8](https://github.com/shustovTE/AVA-AI-Voice-Agent-for-Asterisk/pull/8#discussion_r4227981639),
open **Security / Code scanning**, confirm the rule and location above, and use
**Dismiss alert → False positive** with the following justification (and a link
to the reviewed PR):

> This location publishes only public Silero VAD / Smart Turn ONNX weights.
> Mode 0644 intentionally permits a different container UID to read host-fetched
> models and grants no group/other file write permission. Downloads use private
> temporary files, verify a pinned SHA-256, then publish atomically. Model files
> and parent directories are trusted deployment inputs; existing custom models
> are allowed with a checksum warning. No credentials or call data are written
> here. See SECURITY.md, Public detector model files, and the regression tests.

Dismiss the individual alert, not merely its PR review conversation. GitHub
records the reason/comment and applies the dismissal across branches. Reassess
if a caller starts downloading confidential files or the deployment trust
boundary changes. Adding these comments or merging this documentation does not
itself dismiss the alert.

References: [CodeQL rule](https://codeql.github.com/codeql-query-help/python/py-overly-permissive-file/),
[GitHub alert resolution](https://docs.github.com/en/code-security/how-tos/manage-security-alerts/manage-code-scanning-alerts/resolve-alerts),
[Docker bind mounts](https://docs.docker.com/engine/storage/bind-mounts/).

### 4. Dependency Security

**Automated Scanning** (enabled via CI):
- **Dependabot**: Weekly dependency updates
- **Trivy**: Docker vulnerability scanning
- **CodeQL**: Static code analysis

**Manual Checks**:
```bash
# Check Python dependencies
pip list --outdated

# Scan for known vulnerabilities
pip-audit
```

### 5. Logging Security

**Log Sanitization** (automatic):
- API keys, passwords, tokens automatically redacted in logs
- Example: `api_key: "sk***REDACTED***"`
- Implemented via structlog processor (AAVA-37)

**Log Access Control**:
```bash
# Restrict log file permissions
chmod 640 logs/*.log
chown appuser:appgroup logs/*.log
```

### 6. Production Hardening

**Required for Production**:
- [ ] Change default credentials
- [ ] Enable firewall (ufw/iptables)
- [ ] Configure log rotation
- [ ] Enable monitoring/alerting
- [ ] Regular backup schedule
- [ ] TLS/SSL for external access
- [ ] Rate limiting for API endpoints

**Environment Variables**:
```bash
# Production settings
LOG_LEVEL=info              # Not debug (security risk)
STREAMING_LOG_LEVEL=info    # Not debug (performance impact)
```

---

## Known Security Considerations

### 1. API Provider Dependencies

This application integrates with third-party AI providers:
- **OpenAI**: Processes audio/text via their API
- **Deepgram**: Processes audio via their API
- **Google**: (If configured) Processes audio via their API

**Privacy Implications**:
- Audio is transmitted to cloud providers for processing
- Review provider privacy policies and DPAs
- For complete data privacy, use `local_only` configuration

### 2. Asterisk Integration

**ARI Credentials**:
- Requires Asterisk ARI username/password
- Use dedicated ARI user (not `admin`)
- Grant only necessary permissions
- Example `/etc/asterisk/ari.conf`:
  ```ini
  [AIAgent]
  type=user
  read_only=no
  password=strong_random_password_here
  ```

### 3. Audio File Storage

**Local Hybrid Configuration**:
- Audio files stored in `/mnt/asterisk_media/ai-generated/`
- Contains TTS audio (may include sensitive information)
- Recommendations:
  - Set appropriate filesystem permissions (750)
  - Configure file retention policy
  - Encrypt volume if required by compliance

---

## Compliance

### HIPAA (Healthcare)

If processing Protected Health Information (PHI):
- [ ] Enable audit logging
- [ ] Encrypt audio files at rest
- [ ] Sign Business Associate Agreements (BAAs) with AI providers
- [ ] Implement access controls
- [ ] Configure log retention per requirements

### GDPR (EU Personal Data)

If processing EU personal data:
- [ ] Implement data retention policies
- [ ] Provide data deletion procedures
- [ ] Document data processing activities
- [ ] Obtain necessary consents
- [ ] Review AI provider GDPR compliance

---

## Security Updates

Subscribe to security notifications:
- **GitHub Watch**: Enable "Releases only" notifications
- **Security Advisories**: Check [GitHub Security tab](https://github.com/hkjarral/AVA-AI-Voice-Agent-for-Asterisk/security)
- **Dependabot Alerts**: Review weekly PR updates

---

## Disclosure Policy

### Coordinated Disclosure

We follow coordinated disclosure practices:
1. Reporter notifies us privately
2. We develop and test a fix
3. We release a security patch
4. Public disclosure after patch is available
5. Credit given to reporter (if desired)

### Public Disclosure Timeline

- **Typical**: 30-90 days after initial report
- **May be extended** if fix is complex or requires coordination
- **May be shortened** if exploit is public or actively used

---

## Security Contact

For security-related questions or concerns:
- **Security Reports**: [Your security email]
- **General Security Questions**: Open a GitHub Discussion
- **Emergency**: Tag issue with `security` label (for non-sensitive issues only)

---

## Acknowledgments

We thank the security research community for responsible disclosure practices. Security researchers who have helped improve this project:

- [List of contributors who reported security issues]

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2025-11-07 | 1.0 | Initial security policy |

---

## Additional Resources

- [Production Deployment Guide](docs/PRODUCTION_DEPLOYMENT.md)
- [Configuration Reference](docs/Configuration-Reference.md)
- [Monitoring Guide](docs/MONITORING_GUIDE.md)
