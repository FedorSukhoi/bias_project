from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse


EXCLUDED_PATH_MARKERS = (
    "/cnn-underscored/",
    "/deals/",
    "/entertainment/",
    "/fashion/",
    "/food/",
    "/games/",
    "/gifts/",
    "/lifestyle/",
    "/recipes/",
    "/select/shopping/",
    "/shopping/",
    "/sports/",
    "/style/",
    "/travel/",
)

EXCLUDED_TEXT_PATTERNS = (
    r"\bbest .* deals?\b",
    r"\bdeal of the day\b",
    r"\bgift guide\b",
    r"\bgifts? (?:for|she(?:'|’)ll|he(?:'|’)ll)\b",
    r"\blowest price\b",
    r"\bshop(?:ping)?\b",
    r"\b(?:percent|%) off\b",
)

STRONG_POLITICAL_PATH_MARKERS = (
    "/campaign",
    "/congress",
    "/election",
    "/foreign-policy",
    "/immigration",
    "/national-security",
    "/politics",
    "/senate",
    "/supreme-court",
    "/white-house",
)

CONDITIONAL_POLITICAL_PATH_MARKERS = (
    "/analysis",
    "/commentary",
    "/government",
    "/ideas",
    "/opinion",
    "/policy",
)

# Strong phrases are sufficient by themselves. They describe institutions,
# offices, elections or explicitly political organizations rather than an
# ideological position, so they filter topic without defining the bias label.
STRONG_TEXT_PATTERNS = (
    r"\bballot(?:s|ing)?\b",
    r"\bbipartisan\b",
    r"\bcabinet\b",
    r"\bcampaign(?:s|ing)?\b",
    r"\bcia\b",
    r"\bcongress(?:ional)?\b",
    r"\bconstitution(?:al)?\b",
    r"\bdemocrats?\b",
    r"\bdiplomat(?:s|ic|ically)?\b",
    r"\bdoj\b",
    r"\belection(?:s)?\b",
    r"\belectoral\b",
    r"\bexecutive order\b",
    r"\bfederal government\b",
    r"\bfbi\b",
    r"\bfema\b",
    r"\bforeign policy\b",
    r"\bgeopolitic(?:s|al)\b",
    r"\bgop\b",
    r"\bgovernment shutdown\b",
    r"\bhouse of representatives\b",
    r"\blawmakers?\b",
    r"\bice\b",
    r"\bminister(?:s|ial)?\b",
    r"\bnato\b",
    r"\bparliament(?:ary)?\b",
    r"\bpolitic(?:s|al|ally|ian|ians)\b",
    r"\bprime minister\b",
    r"\brepublicans?\b",
    r"\bsecretary of state\b",
    r"\bsecret service\b",
    r"\bsenat(?:e|or|ors|orial)\b",
    r"\bsupreme court\b",
    r"\bthe administration\b",
    r"\bunited nations\b",
    r"\bvot(?:e|es|ed|ing)\b",
    r"\bvoters?\b",
    r"\bwhite house\b",
)

# Two or more supporting matches are required when no strong signal exists.
SUPPORTING_TEXT_PATTERNS = (
    r"\babortion\b",
    r"\badministration\b",
    r"\bborder\b",
    r"\bceasefire\b",
    r"\bclimate (?:law|policy|regulation)\b",
    r"\battorneys? general\b",
    r"\bcontent moderation\b",
    r"\bcourt\b",
    r"\bgovernor\b",
    r"\bimmigration\b",
    r"\bjudge\b",
    r"\bjustice department\b",
    r"\blaw\b",
    r"\blegislat(?:ion|ive|ure)\b",
    r"\bmayor\b",
    r"\bmilitary\b",
    r"\bpresident(?:ial)?\b",
    r"\bregulat(?:ion|or|ory)\b",
    r"\bsanctions?\b",
    r"\btariffs?\b",
    r"\btax(?:es|ation)?\b",
    r"\btreaty\b",
    r"\bwar\b",
)


@dataclass(frozen=True)
class RelevanceResult:
    relevant: bool
    score: int
    reason: str


def _matches(patterns: tuple[str, ...], text: str) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE)]


def assess_political_relevance(headline: str, summary: str, url: str) -> RelevanceResult:
    path = urlparse(url).path.casefold()
    # Some feeds put the entire article body in their summary field. Limiting
    # context avoids classifying a culture or shopping story as political just
    # because a political word occurs thousands of characters later.
    text = f"{headline}\n{summary[:800]}".strip()

    for marker in EXCLUDED_PATH_MARKERS:
        if marker in path:
            return RelevanceResult(False, 0, f"excluded_path:{marker.strip('/')}")
    excluded_text = _matches(EXCLUDED_TEXT_PATTERNS, headline)
    if excluded_text:
        return RelevanceResult(False, 0, "excluded_commerce_language")

    strong = _matches(STRONG_TEXT_PATTERNS, text)
    supporting = _matches(SUPPORTING_TEXT_PATTERNS, text)

    for marker in STRONG_POLITICAL_PATH_MARKERS:
        if marker in path:
            return RelevanceResult(True, 3, f"political_path:{marker.strip('/')}")
    for marker in CONDITIONAL_POLITICAL_PATH_MARKERS:
        if marker in path and (strong or supporting):
            return RelevanceResult(True, 2, f"political_context_path:{marker.strip('/')}")

    if strong:
        return RelevanceResult(True, 2 + min(len(strong) - 1, 2), "strong_political_language")

    if len(supporting) >= 2:
        return RelevanceResult(True, 2, "multiple_political_signals")
    if len(supporting) == 1:
        return RelevanceResult(False, 1, "single_ambiguous_signal")
    return RelevanceResult(False, 0, "no_political_signal")
