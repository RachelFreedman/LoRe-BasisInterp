# concept_library_v2.py
#
# Extended concept library for the synthetic persona dataset.
#
# Motivation: the original 11-concept library has an effective rank of 3.48. The five quality
# concepts (helpfulness, factuality, safety, values, sycophancy) have an effective rank of 1.54
# between them -- safety and values sit at cos 0.98. Personas built over those axes would be
# indistinguishable by construction, so the control could fail for reasons unrelated to the
# method under test.
#
# BASE: the six stylistic concepts that survived the collinearity screen (effective rank 3.996).
#       Definitions are copied verbatim from concept_library.py so that the only difference
#       between concept_vectors.pt and concept_vectors_v2.pt is the generating model. That makes
#       the pair a direct measurement of how much generator identity moves a concept vector.
#
#       Note on 'diversity': it sits at cos 0.84-0.89 from every member of the quality cluster and
#       is the most quality-aligned direction in the surviving set. It serves as the quality axis
#       here. The name understates what the vector measures.
#
# CANDIDATES: six new stylistic contrasts, to be screened on effective rank. Two candidates from
#       the earlier plan were cut before generation: 'hedging' is 'confidence' inverted, and
#       'directness' overlaps both 'confidence' and 'verbosity'.
#
# DESIGN CONSTRAINT for candidates: neither pole may be the worse response. Every concept whose
#       low pole reads as "bad" collapses onto the quality axis, which is precisely how the
#       original library lost its rank. Both poles must be defensible stylistic choices that
#       different readers could genuinely prefer.

BASE_CONCEPTS = {
    "fluency": {
        "description": "The response is grammatically correct, well-structured, and easy to read",
        "high": "Write a response that is grammatically flawless, well-structured, and flows naturally",
        "low":  "Write a response with awkward phrasing, grammatical errors, and poor sentence structure",
    },
    "diversity": {
        "description": "The response offers varied perspectives or multiple approaches",
        "high": "Write a response that presents multiple perspectives, approaches, or viewpoints",
        "low":  "Write a response that presents only one narrow perspective with no alternatives",
    },
    "creativity": {
        "description": "The response shows originality and novel thinking",
        "high": "Write a response that is creative, original, and offers unexpected or novel ideas",
        "low":  "Write a response that is generic, formulaic, and completely predictable",
    },
    "confidence": {
        "description": "The response expresses certainty and authority in its claims",
        "high": "Write a response that is assertive, confident, and speaks with authority",
        "low":  "Write a response that is hesitant, hedging, and full of qualifiers like 'maybe' or 'I think'",
    },
    "formatting": {
        "description": "The response uses markdown, bullet points, headers, and structured layout",
        "high": "Write a response using rich formatting: headers, bullet points, bold text, and clear visual structure",
        "low":  "Write a response as a plain, unformatted wall of text with no structural elements",
    },
    "repetition": {
        "description": "The response contains repetitive text patterns or restates the same points",
        "high": "Write a response that repeats the same ideas multiple times in different words, restating key points redundantly",
        "low":  "Write a response where every sentence adds new information with no redundancy",
    },
}

CANDIDATE_CONCEPTS = {
    "verbosity": {
        "description": "The response is long and elaborated rather than compressed and terse",
        "high": "Write a long, elaborated response that develops the answer at length, with context, background and worked-through explanation",
        "low":  "Write a terse, compressed response that states the answer in as few words as possible and stops",
    },
    "formality": {
        "description": "The response uses a formal, academic register rather than a casual, conversational one",
        "high": "Write in a formal academic register: complete sentences, no contractions, measured professional tone, third person where natural",
        "low":  "Write in a casual conversational register: contractions, everyday words, the tone of talking to a friend",
    },
    "concreteness": {
        "description": "The response grounds its points in specific examples, numbers and cases rather than general statements",
        "high": "Write a response grounded in specifics: concrete examples, actual numbers, named cases, and particular scenarios",
        "low":  "Write a response that stays at the level of general principles and abstract statements, without specific examples or figures",
    },
    "technicality": {
        "description": "The response uses domain-specific technical vocabulary rather than plain-language framing",
        "high": "Write for a domain expert: use precise technical terminology and assume specialist background knowledge",
        "low":  "Write for an intelligent non-specialist: use plain language and everyday analogies, avoiding technical vocabulary",
    },
    "warmth": {
        "description": "The response has a personal, empathetic tone rather than a neutral, clinical one",
        "high": "Write with a warm personal tone that acknowledges the reader's situation and feelings",
        "low":  "Write in a neutral clinical tone that addresses the question impersonally, with no acknowledgement of the reader",
    },
    "humor": {
        "description": "The response is playful and light rather than earnest and straight-faced",
        "high": "Write with a playful light touch: wit, wordplay, or an amusing aside where it fits naturally",
        "low":  "Write in an earnest, straight-faced manner with no humour or playfulness of any kind",
    },
}

CONCEPT_LIBRARY_V2 = {**BASE_CONCEPTS, **CANDIDATE_CONCEPTS}
