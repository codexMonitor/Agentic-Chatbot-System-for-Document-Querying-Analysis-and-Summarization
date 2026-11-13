  | Date | Change | Hypothesis | Metrics before | Metrics after | Decision |
  |---|---|---|---|---|---|
  | 2026-11-13 | k: 4 → 8 | More context improves answers | key facts: answer 0.936, context 0.962; 4.9 chunks | same scores; 9.7 chunks (about 2x context) | revert |
  | 2026-11-13 | chunk size: 500 → 800 | Bigger chunks keep facts together | key facts: answer 0.936; about 2,400 context chars | answer 0.962 (+1 fact, within noise); about 3,900 chars (+60%) | revert |
  | 2026-11-13 | Answer in the question’s language (prompt fix) | Stops language flips | row 06 answered in German (k=8 run) | row 06 in English; key facts unchanged	 | keep |
