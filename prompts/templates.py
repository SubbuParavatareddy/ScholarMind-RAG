"""All LLM prompt templates and constants used across the application."""

# ── RAG / Chat ──────────────────────────────────────────────────────────────────

RAG_TEMPLATE = """You are ScholarMind, a precise AI research assistant.
Answer using ONLY the retrieved context. Be structured and cite evidence.
If context is insufficient, say so — never fabricate.

Context:
{context}

Question: {question}

Answer:"""

FOLLOW_UP_TEMPLATE = """Suggest 3 follow-up questions. Return ONLY JSON array of 3 strings.
Q: {q}
A: {a}
JSON:"""

HEDGE_PHRASES = {"don't have", "not enough", "cannot", "no information", "not provided"}

# ── Analysis ────────────────────────────────────────────────────────────────────

SUMMARY_PROMPT = """Analyze this research paper and return a JSON object with keys:
"one_liner","problem","method","results","contribution","limitations",
"keywords"(array 8-10),"paper_type"(empirical/theoretical/survey/system/position),
"venue_guess","year_guess".
Return ONLY valid JSON, no markdown fences.

Excerpt:\n{ctx}"""

COMPARE_PROMPT = """Compare two papers. Return JSON:
"a_contribution","b_contribution","similarities"(array 3),"differences"(array 4),
"complementary","read_order","a_strength","b_strength".
Return ONLY valid JSON.

Paper A ({na}):\n{ca}\n\nPaper B ({nb}):\n{cb}"""

FLASHCARD_PROMPT = """Make 6 study flashcards. Return JSON array of {{"q":...,"a":...}}.
Return ONLY valid JSON array, no markdown fences.

Paper:\n{ctx}"""

GAP_PROMPT = """Find research gaps. Return JSON:
"explicit_gaps"(array 3-4),"implicit_gaps"(array 3-4),
"future_directions"(array 3-4),"open_questions"(array 3-4).
Return ONLY valid JSON.

Paper:\n{ctx}"""

ELI15_PROMPT = "Explain this paper to a curious 15-year-old. Max 180 words.\n\nPaper:\n{ctx}"
