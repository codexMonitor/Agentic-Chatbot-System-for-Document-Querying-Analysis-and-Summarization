  | Date | Change | Hypothesis | Metrics before | Metrics after | Decision |
  |---|---|---|---|---|---|
  | 2026-11-13 | k: 4 → 8 | More context improves answers | intent 0.857, hit 1.0, 4.9 chunks | intent 0.857, hit 1.0, 9.7 chunks (about 2x context), no quality gain | revert |
  | 2026-11-13 | chunk size: 500 → 800 | Bigger chunks keep facts together | same baseline, about 2,400 context chars | same quality, about 3,900 context chars (+60%) | revert |