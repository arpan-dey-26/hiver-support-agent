"""All prompts, versioned in one place so the report can cite exact wording.

Changing any string here means bumping its version, because the response cache
is keyed on prompt text and old cached results would no longer correspond.
"""

TAXONOMY_DISCOVERY_V1 = """\
You are helping design an intent taxonomy for a customer-support agent that
handles public Twitter messages sent to {brand}.

Below are {n_clusters} groups of real customer messages, produced by clustering.
For each group you see its most distinctive terms and a sample of real messages.

{clusters}

Propose a SMALL taxonomy of {lo}-{hi} intents that covers most of this traffic.

Rules:
- Every intent must be grounded in the groups above, not in general knowledge
  about this industry.
- Merge groups that are the same underlying request.
- Each intent needs a crisp one-line definition, an inclusion rule and an
  exclusion rule, so two people would label the same message the same way.
- Mark each intent `account_specific: true` if answering it normally requires
  looking at one customer's account, `false` if it can be answered from published
  policy or general troubleshooting.
- Do NOT include a catch-all: an "other" label is added automatically.

Return ONLY JSON:
{{"intents": [{{"name": "snake_case_name", "definition": "...",
  "includes": "...", "excludes": "...", "account_specific": true,
  "example_messages": ["...", "..."]}}]}}
"""

CLASSIFY_V1 = """\
Classify the customer support message below into exactly one intent.

Intents for {brand}:
{taxonomy}
- other: the message fits none of the above, is off-topic, is only praise or
  complaint with no request, or is too vague to classify.

Message:
\"\"\"{message}\"\"\"

Return ONLY JSON:
{{"intent": "<one intent name>", "confidence": <0.0-1.0>,
  "runner_up": "<second most likely intent name>",
  "multi_intent": <true if the message contains two distinct requests>}}
"""

CLASSIFY_BATCH_V1 = """\
Classify each customer support message into exactly one intent.

Intents for {brand}:
{taxonomy}
- other: fits none of the above, off-topic, praise/complaint with no request, or
  too vague.

Messages:
{messages}

Return ONLY JSON:
{{"results": [{{"id": <int>, "intent": "<name>", "confidence": <0.0-1.0>,
  "runner_up": "<name>", "multi_intent": <bool>}}]}}
"""

GENERATE_V1 = """\
You draft replies for {brand}'s customer support team on Twitter. A human agent
reviews every draft before it is sent, so accuracy matters more than confidence.

Customer message:
\"\"\"{message}\"\"\"

Classified intent: {intent}

Historical precedents — real past exchanges where {brand} answered a similar
message. These are your ONLY source of factual claims:

{evidence}

Rules:
- Every factual claim in your reply must be traceable to one of the precedents above.
- If the precedents do not support an answer, say so and set sufficient_evidence to false.
- Never promise a refund, credit, account change, or any specific action on the
  customer's account. You cannot perform actions.
- Do not invent policies, prices, timeframes, URLs or contact details that do not
  appear in the precedents.
- Match the tone of the precedents. Twitter length: at most 2 short sentences.

Return ONLY JSON:
{{"reply": "<the draft>",
  "sufficient_evidence": <true|false>,
  "evidence_used": [<precedent numbers you relied on, e.g. 1, 3>],
  "claims": [{{"claim": "<a factual assertion in your reply>",
               "supported_by": [<precedent numbers, empty list if none>]}}]}}
"""

ANALYZE_V1 = """\
You handle customer support for {brand} on Twitter. A human agent reviews every
draft before it is sent, so accuracy matters more than confidence.

Do two things in ONE response: classify the message, and draft a reply.

Intents:
{taxonomy}
- other: fits none of the above, off-topic, praise/complaint with no request, or
  too vague to classify.

Customer message:
\"\"\"{message}\"\"\"

Historical precedents — real past exchanges where {brand} answered a similar
message. These are your ONLY permitted source of factual claims:

{evidence}

Rules for the draft:
- Every factual claim must be traceable to a precedent above.
- If the precedents do not support an answer, say so and set sufficient_evidence false.
- Never promise a refund, credit, account change or any action on the customer's
  account. You cannot perform actions.
- Do not invent policies, prices, timeframes, URLs or contact details.
- Match the tone of the precedents. At most 2 short sentences.

Return ONLY JSON:
{{"intent": "<one intent name>", "confidence": <0.0-1.0>,
  "runner_up": "<second most likely intent>",
  "multi_intent": <true if two distinct requests>,
  "reply": "<the draft>",
  "sufficient_evidence": <true|false>,
  "evidence_used": [<precedent numbers relied on>],
  "claims": [{{"claim": "<a factual assertion in your reply>",
               "supported_by": [<precedent numbers, empty if none>]}}]}}
"""

JUDGE_V1 = """\
You are evaluating a DRAFT customer-support reply. Be strict and specific.

Customer message:
\"\"\"{message}\"\"\"

Evidence the drafter was given (its only permitted source of facts):
{evidence}

Draft reply:
\"\"\"{reply}\"\"\"

The system classified the intent as "{intent}" and decided to "{decision}".

Score each dimension 1-5. Write the one-sentence justification BEFORE the score.

- relevance: does it address what the customer actually asked?
  1 = ignores the request, 5 = addresses exactly what was asked
- groundedness: is every claim traceable to the evidence above?
  1 = contradicts or ignores evidence, 5 = every claim traceable
- helpfulness: does the customer know what happens next?
  1 = no actionable next step, 5 = completely clear next step
- tone_fit: does it match the register of the evidence replies?
  1 = off-brand, robotic or rude, 5 = matches well
- completeness: is the main ask fully addressed?
  1 = main ask unanswered, 5 = fully addressed

Then two binary judgements:
- unsupported_claims: 1 if the reply asserts ANY fact not supported by the
  evidence (invented policy, price, timeframe, URL, or promise of action), else 0.
- escalation_appropriate: 1 if "{decision}" was the right call for this message,
  else 0. A message needing account access, a refund, or involving legal, safety
  or financial-harm content should be escalated.

Return ONLY JSON:
{{"relevance": {{"why": "...", "score": <1-5>}},
  "groundedness": {{"why": "...", "score": <1-5>}},
  "helpfulness": {{"why": "...", "score": <1-5>}},
  "tone_fit": {{"why": "...", "score": <1-5>}},
  "completeness": {{"why": "...", "score": <1-5>}},
  "unsupported_claims": {{"why": "...", "flag": <0|1>}},
  "escalation_appropriate": {{"why": "...", "flag": <0|1>}}}}
"""

PRE_ANNOTATE_V1 = CLASSIFY_BATCH_V1

VERSIONS = {
    "taxonomy_discovery": "v1",
    "classify": "v1",
    "classify_batch": "v1",
    "generate": "v1",
    "analyze": "v1",
    "judge": "v1",
    "pre_annotate": "v1",
}
