# Security Policy

## Supported versions

Until the first stable release, only the latest tagged version receives
security fixes.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting for this repository. Include the
affected version, impact, reproduction steps, and any suggested mitigation.
Please do not disclose the issue publicly before a fix is available.

## Deployment boundary

Token Miser does not authenticate inbound proxy requests. The default loopback
binding is intentional. Deployments that listen on another interface must add
an authenticated gateway or an equivalent trusted network boundary.
