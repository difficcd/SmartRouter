# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0
#
# Prints the CPU package temperature in Celsius, or nothing at all.
#
# Reads LibreHardwareMonitor's local web server (Options > Remote Web Server >
# Run). Printing nothing when it cannot be read is deliberate: the callers
# treat "no reading" as "fall back to a fixed cooldown" rather than inventing a
# number. Guessing the temperature is exactly the mistake this replaces -- an
# earlier session reported "no throttling, safe" from a CPU-utilisation proxy
# while the machine was actually at 95C.

try {
    $r = Invoke-WebRequest "http://localhost:8085/data.json" -TimeoutSec 4 -UseBasicParsing -ErrorAction Stop
    $json = $r.Content | ConvertFrom-Json
} catch { exit 1 }

$best = $null
function Walk($node) {
    if ($node.Text -eq "CPU Package" -and $node.Value -match '^\s*([0-9.]+)') {
        $script:best = [double]$Matches[1]
    }
    # Fall back to the hottest core if this build has no package sensor.
    if ($null -eq $script:best -and $node.Text -eq "Core Max" -and $node.Value -match '^\s*([0-9.]+)') {
        $script:best = [double]$Matches[1]
    }
    foreach ($c in $node.Children) { Walk $c }
}
Walk $json

if ($null -eq $best) { exit 1 }
"{0:N1}" -f $best
