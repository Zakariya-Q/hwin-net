# Security Policy

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in HWIN-Net, please report it
responsibly by emailing us at [INSERT SECURITY EMAIL] instead of creating a
public issue.

We will:
1. Acknowledge receipt within 48 hours
2. Provide an initial assessment within 5 business days
3. Keep you informed of our progress
4. Credit you in the fix (if desired)

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fixes

## Security Considerations

HWIN-Net is a research implementation. Consider:

* **No production hardening**: Not hardened for production deployment
* **Data privacy**: Ensure compliance with local regulations when processing water quality data
* **Model artifacts**: Checkpoint files may contain training data statistics
* **Dependencies**: Regularly update dependencies for security patches

## Dependency Security

We use GitHub Dependabot for automated dependency updates. Security advisories
are monitored and addressed promptly.

## Disclosure Policy

We follow responsible disclosure. Vulnerabilities will be:
1. Privately disclosed to maintainers
2. Fixed in a timely manner
3. Publicly disclosed after fix is available
4. Credited to reporter (unless anonymity requested)
