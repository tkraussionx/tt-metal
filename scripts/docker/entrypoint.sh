#!/bin/bash

set -euo pipefail

# ccache doesn't support TLS; create a TLS tunnel
stunnel

# Keep the container alive, with minimal load
tail -f /dev/null
