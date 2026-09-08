"""
text_corpus.py — procedural short-text corpus generator.

No downloads, no external corpus files — fully deterministic given a seed.
Ten topic domains x templated combinations give semantically clustered
short sentences, which is what we want for a meaningful semantic-search
demo in Milestone 4 (real embedding geometry is lumpy; this mimics that
by construction instead of hoping random noise looks lumpy).
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple

_DOMAINS = {
    "weather": dict(
        subject=["A cold front", "Heavy rain", "The morning fog", "A sudden storm",
                 "Clear skies", "Strong winds", "A heatwave", "Light snow",
                 "Humid air", "A thunderstorm", "The afternoon sun", "A cool breeze"],
        verb=["is moving across", "settled over", "swept through", "is expected in",
              "lingered above", "rolled into", "developed over", "cleared out of",
              "built up near", "drifted toward"],
        object=["the coastal region", "downtown", "the valley", "the northern plains",
                "the mountain pass", "the harbor", "the airport", "the suburbs",
                "the desert basin", "the lake district"],
    ),
    "sports": dict(
        subject=["The home team", "A rookie forward", "The veteran coach", "The visiting squad",
                 "An injured striker", "The star pitcher", "The defense", "A young sprinter",
                 "The championship team", "A substitute goalkeeper"],
        verb=["dominated", "struggled against", "narrowly beat", "was eliminated by",
              "trained hard for", "signed a new deal with", "clashed with", "outscored",
              "prepared all season for", "upset"],
        object=["the league finals", "their biggest rival", "the season opener",
                "the playoff bracket", "a tough opponent", "the regional tournament",
                "the away game", "the title match", "a sold-out crowd", "the final round"],
    ),
    "food": dict(
        subject=["The chef", "A local bakery", "The new restaurant", "A street vendor",
                 "The pastry cook", "A home cook", "The food critic", "A catering team",
                 "The farmers market", "A late-night diner"],
        verb=["prepared", "served", "experimented with", "perfected", "sold out of",
              "reinvented", "garnished", "slow-cooked", "plated", "grilled"],
        object=["a spicy noodle dish", "fresh sourdough bread", "a seasonal vegetable stew",
                "grilled seafood", "a rich chocolate dessert", "handmade dumplings",
                "a citrus salad", "smoked barbecue ribs", "a creamy risotto",
                "roasted root vegetables"],
    ),
    "travel": dict(
        subject=["A group of tourists", "The tour guide", "A solo backpacker", "The cruise ship",
                 "A family on vacation", "The travel agency", "A photographer", "The airline crew",
                 "A pair of hikers", "The exchange student"],
        verb=["explored", "wandered through", "booked a trip to", "photographed",
              "got lost in", "recommended", "flew into", "hiked across", "toured", "camped near"],
        object=["the old town square", "a remote mountain village", "the coastal cliffs",
                "a bustling night market", "an ancient temple", "the national park",
                "a quiet fishing town", "the botanical gardens", "a crowded train station",
                "the desert dunes"],
    ),
    "technology": dict(
        subject=["The startup", "A software engineer", "The research lab", "A new smartphone",
                 "The open-source project", "A hardware team", "The cybersecurity firm",
                 "An AI model", "The cloud provider", "A robotics company"],
        verb=["launched", "patched a bug in", "open-sourced", "benchmarked", "redesigned",
              "shipped an update to", "optimized", "deprecated", "scaled up", "reverse-engineered"],
        object=["a new mobile app", "the authentication system", "a machine learning pipeline",
                "the user interface", "a distributed database", "the recommendation engine",
                "an internal tool", "the payment gateway", "a computer vision model",
                "the deployment pipeline"],
    ),
    "finance": dict(
        subject=["The central bank", "An investment firm", "The stock market", "A startup founder",
                 "The regulator", "A hedge fund", "The board of directors", "A retail investor",
                 "The credit union", "An economist"],
        verb=["raised concerns about", "forecasted", "adjusted", "reported", "reviewed",
              "restructured", "invested heavily in", "downgraded", "audited", "hedged against"],
        object=["quarterly earnings", "interest rate policy", "a merger deal", "market volatility",
                "a new funding round", "consumer spending trends", "the annual budget",
                "a risk assessment", "currency fluctuations", "a pension fund"],
    ),
    "health": dict(
        subject=["The clinic", "A physical therapist", "The research team", "A marathon runner",
                 "The nutritionist", "A new patient", "The hospital staff", "A yoga instructor",
                 "The public health department", "A personal trainer"],
        verb=["recommended", "monitored", "treated", "studied the effects of", "designed a plan for",
              "screened for", "published findings on", "adjusted", "tracked progress on", "advised against"],
        object=["a sleep routine", "a knee injury", "a balanced diet", "cardiovascular health",
                "a training regimen", "seasonal allergies", "recovery exercises",
                "stress management techniques", "a vaccination schedule", "muscle recovery"],
    ),
    "movies": dict(
        subject=["The director", "A film critic", "The lead actor", "An indie studio",
                 "The screenwriter", "A film festival", "The production crew", "A cinematographer",
                 "The animation team", "A first-time filmmaker"],
        verb=["premiered", "reviewed", "reshot", "storyboarded", "scored", "edited",
              "cast the lead role for", "adapted", "shot on location for", "wrapped filming on"],
        object=["a period drama", "an animated feature", "a low-budget thriller",
                "a documentary series", "a sci-fi trilogy", "an award-winning short film",
                "a romantic comedy", "a war epic", "a mystery series", "a coming-of-age story"],
    ),
    "science": dict(
        subject=["The astrophysicist", "A field biologist", "The research institute", "A geologist",
                 "The lab technician", "A climate scientist", "The genetics team", "An oceanographer",
                 "The particle physics group", "A graduate student"],
        verb=["discovered", "measured", "modeled", "published a study on", "collected samples of",
              "simulated", "observed", "cataloged", "analyzed", "peer-reviewed"],
        object=["a distant exoplanet", "coral reef degradation", "seismic activity",
                "a new bacterial strain", "atmospheric carbon levels", "deep-sea vents",
                "particle collision data", "glacial melt rates", "a genetic mutation",
                "migratory bird patterns"],
    ),
    "home": dict(
        subject=["The homeowner", "A contractor", "The interior designer", "A DIY enthusiast",
                 "The landscaping crew", "A plumber", "The electrician", "A real estate agent",
                 "The renovation team", "A first-time buyer"],
        verb=["remodeled", "repainted", "rewired", "landscaped", "renovated", "inspected",
              "installed new fixtures in", "refinished", "waterproofed", "insulated"],
        object=["the kitchen", "the backyard patio", "a small apartment", "the basement",
                "the upstairs bathroom", "an old farmhouse", "the garage", "the front porch",
                "a rental property", "the attic"],
    ),
}


def _domain_sentences(vocab: dict, count: int, rng: np.random.Generator) -> List[str]:
    subs, verbs, objs = vocab["subject"], vocab["verb"], vocab["object"]
    combos = [(i, j, l) for i in range(len(subs)) for j in range(len(verbs)) for l in range(len(objs))]
    if count > len(combos):
        raise ValueError("not enough unique combinations for requested count")
    chosen = rng.permutation(len(combos))[:count]
    return [f"{subs[combos[c][0]]} {verbs[combos[c][1]]} {objs[combos[c][2]]}." for c in chosen]


def generate_text_corpus(n: int = 5000, seed: int = 42) -> Tuple[List[str], List[str]]:
    """Returns (texts, domain_labels), both length n."""
    rng = np.random.default_rng(seed)
    domain_names = list(_DOMAINS.keys())
    n_domains = len(domain_names)
    base, extra = divmod(n, n_domains)

    texts: List[str] = []
    labels: List[str] = []
    for i, name in enumerate(domain_names):
        count = base + (1 if i < extra else 0)
        sentences = _domain_sentences(_DOMAINS[name], count, rng)
        texts.extend(sentences)
        labels.extend([name] * count)

    perm = rng.permutation(len(texts))
    texts = [texts[i] for i in perm]
    labels = [labels[i] for i in perm]
    return texts, labels


if __name__ == "__main__":
    texts, labels = generate_text_corpus(5000, seed=42)
    print(f"generated {len(texts)} texts across {len(set(labels))} domains")
    for t in texts[:5]:
        print(" -", t)
