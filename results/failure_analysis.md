# Failure analysis

Mined from real run output; every example carries its tweet id.

### 1. over-escalation  — 74 of 150 (49.3%)

- `tweet_id 565509` — human said auto-handle; system escalated (unknown_intent). Message: "@hulu_support I would definitely check that out. I appreciate your response. Wondering such repetition not going to impress customers though"
- `tweet_id 2927259` — human said auto-handle; system escalated (conflicting_precedents). Message: "@hulu_support Firestick. We've been experiencing playback failures and buffering/pixelization issues since the last push."

**Hypothesis:** the allowlist is deliberately conservative and the evidence floor is set at the DEV 25th percentile, so thin-but-adequate evidence still escalates. This is the intended direction of the trade-off.


### 2. poorly grounded reply  — 48 of 150 (32.0%)

- `tweet_id 559291` — judge groundedness 2: The evidence consistently shows the brand asking for device information and providing a troubleshooting link, whereas the draft promises an escalation that is n
- `tweet_id 464786` — judge groundedness 2: The reply promises to pass the issue to a team member to 'look into it properly,' which is not supported by the evidence where similar feedback is simply shared

**Hypothesis:** retrieved precedents were topically related but did not answer the question, so the model fell back on general knowledge.


### 3. no historical precedent retrieved  — 46 of 150 (30.7%)

- `tweet_id 460328` — BM25 returned nothing above the floor. Message: "@hulu_support Can you add @224379?"
- `tweet_id 458827` — BM25 returned nothing above the floor. Message: "@hulu_support Nvmmmm I managed to get it to workkkk"

**Hypothesis:** BM25 needs lexical overlap; messages using different vocabulary from the historical corpus retrieve nothing. This is the known cost of the lexical-only retrieval choice.


### 4. unsupported claim in draft  — 45 of 150 (30.0%)

- `tweet_id 559291` — judge flagged an unsupported claim. Reply: "Oh no! We'd be upset too. What device is in use? Let's start with: https://t.co/HXpLyTU79x."
- `tweet_id 464786` — judge flagged an unsupported claim. Reply: "Thank you so much for the insight. We'll share it with the team. Feel free to leave feedback here: https://t.co/n4qDfJdbFR"

**Hypothesis:** the model generalises from a precedent rather than restating it, producing plausible specifics (timeframes, policies) the evidence does not contain.


### 5. model output failed to parse  — 11 of 150 (7.3%)

- `tweet_id 458827` — structured-JSON parse failure
- `tweet_id 2901719` — structured-JSON parse failure

**Hypothesis:** structured-output failures under load or on unusual inputs.
