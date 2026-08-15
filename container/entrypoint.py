# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0

"""Container entry point for one-tier prompt routing.

Uses router_main (trained predictor + budget-constrained allocation, added
to heuristic.py -- see the "difficcd team router additions" section in that
file) instead of the reference `main` (weak baseline). Same --input/--tier
/--output contract either way (docs/RUNTIME.md), so nothing else changes.
"""

from __future__ import annotations

from ossp_router.heuristic import router_main


if __name__ == "__main__":
    raise SystemExit(router_main())
