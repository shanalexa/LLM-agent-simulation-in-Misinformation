# Prompt templates and experiment texts are loaded from config.yaml via config.init_config().
# Values below are fallbacks before init_config() runs.

IDEOLOGY_LEVEL_NAMES = {
    1: "Very liberal",
    2: "Liberal",
    3: "Slightly liberal",
    4: "Moderate / center",
    5: "Slightly conservative",
    6: "Conservative",
    7: "Very conservative",
}

DEFAULT_EXPERIMENT_RUMOR_TOPIC = (
    "Public Safety Report: Federal crime statistics have confirmed what many Americans have long suspected — "
    "undocumented immigrants commit violent crimes at nearly twice the rate of native-born citizens, "
    "representing an escalating threat to national security. The safety of American families is at serious and growing risk."
)

LTM_MAX_WORDS = 300

CHECKPOINT_DAY = 7
INTERVENTION_DAY = 8

EXPERIMENT_BRANCHES = ("NYT", "WSJ", "AI", "Neutral", "Control")

REFUTATION_TEXTS = {
    "NYT": (
        "The New York Times, a liberal-leaning major news outlet, has published a fact-check report "
        "concluding that the claim regarding undocumented immigrant crime rates being twice those of "
        "native-born citizens is misleading and not supported by available evidence. According to "
        "peer-reviewed criminological research and official federal crime data, undocumented immigrants "
        "have substantially lower crime rates than native-born citizens across a range of felony offenses "
        "between 2012 and 2018：Relative to undocumented immigrants, U.S.-born citizens are over 2 times "
        "more likely to be arrested for violent crimes, 2.5 times more likely to be arrested for drug crimes, "
        "and over 4 times more likely to be arrested for property crimes. Recent federal data also reveals "
        "that the incarceration rate for native-born citizens (1,195 per 100,000) is significantly higher than "
        "that for undocumented immigrants (674 per 100,000). The New York Times concludes that the claim "
        "framing undocumented immigrants as a disproportionate public safety threat is not supported by "
        "available evidence."
    ),
    "WSJ": (
        "The Wall Street Journal, a conservative-leaning major news outlet, has published a fact-check report "
        "concluding that the claim regarding undocumented immigrant crime rates being twice those of "
        "native-born citizens is misleading and not supported by available evidence. According to "
        "peer-reviewed criminological research and official federal crime data, undocumented immigrants "
        "have substantially lower crime rates than native-born citizens across a range of felony offenses "
        "between 2012 and 2018：Relative to undocumented immigrants, U.S.-born citizens are over 2 times "
        "more likely to be arrested for violent crimes, 2.5 times more likely to be arrested for drug crimes, "
        "and over 4 times more likely to be arrested for property crimes. Recent federal data also reveals "
        "that the incarceration rate for native-born citizens (1,195 per 100,000) is significantly higher than "
        "that for undocumented immigrants (674 per 100,000). The Wall Street Journal concludes that the claim "
        "framing undocumented immigrants as a disproportionate public safety threat is not supported by "
        "available evidence."
    ),
    "Neutral": (
        "PolitiFact, an independent non-partisan fact-checking organization, has published a fact-check report "
        "concluding that the claim regarding undocumented immigrant crime rates being twice those of "
        "native-born citizens is misleading and not supported by available evidence. According to "
        "peer-reviewed criminological research and official federal crime data, undocumented immigrants "
        "have substantially lower crime rates than native-born citizens across a range of felony offenses "
        "between 2012 and 2018：Relative to undocumented immigrants, U.S.-born citizens are over 2 times "
        "more likely to be arrested for violent crimes, 2.5 times more likely to be arrested for drug crimes, "
        "and over 4 times more likely to be arrested for property crimes. Recent federal data also reveals "
        "that the incarceration rate for native-born citizens (1,195 per 100,000) is significantly higher than "
        "that for undocumented immigrants (674 per 100,000). PolitiFact concludes that the claim framing "
        "undocumented immigrants as a disproportionate public safety threat is not supported by available evidence."
    ),
    "AI": (
        "An automated algorithmic fact-checking system, produced a report by systematically processing "
        "multiple peer-reviewed studies and government datasets without human editorial input or institutional "
        "affiliation. This report concludes that the claim regarding undocumented immigrant crime rates being "
        "twice those of native-born citizens is misleading and not supported by available evidence. According "
        "to peer-reviewed criminological research and official federal crime data, undocumented immigrants "
        "have substantially lower crime rates than native-born citizens across a range of felony offenses "
        "between 2012 and 2018：Relative to undocumented immigrants, U.S.-born citizens are over 2 times "
        "more likely to be arrested for violent crimes, 2.5 times more likely to be arrested for drug crimes, "
        "and over 4 times more likely to be arrested for property crimes. Recent federal data also reveals "
        "that the incarceration rate for native-born citizens (1,195 per 100,000) is significantly higher than "
        "that for undocumented immigrants (674 per 100,000). The algorithm-generated report concludes that "
        "the claim framing undocumented immigrants as a disproportionate public safety threat is not supported "
        "by available evidence."
    ),
    "Control": "",
}

day0_cognitive_prompt = """You are simulating a real person with the following persona:
{persona_prompt}

Read the following breaking news:
{misinformation_text}

Based on your persona, analyze whether you believe this information. Provide your internal reasoning, and then compose a tweet expressing your opinion. Finally, rate your belief on a 7-point Likert scale (1 = Strong Disbelief, 7 = Strong Belief).

Output exactly in this JSON format, and nothing else:
{{
"belief_score": <integer 1-7>,
"opinion": "<tweet>",
"reasoning": "<first-person internal monologue>"
}}
"""

day0_ltm_init_prompt = """Summarize the news you received and your internal reasoning. Compress this summary into a concise long-term memory block of no more than {max_words} words to retain core beliefs while saving cognitive space. Return only the compressed long-term memory text, with no preamble."""

daily_stm_prompt = """Summarize the following 3 opinions you have heard from other agents in a few sentences, explicitly including their belief score about the news.

Peer records (tweet + belief_score):
{peer_records}

Return only the short-term memory summary."""

daily_cognitive_prompt = """You are simulating a real person with the following persona:
{persona_prompt}

Your overarching mindset is shaped by your long-term memory:
{long_term_memory}

Yesterday, your public stance was:
{previous_day_tweet}

Today, you observed the following from your peers (Short-Term Memory):
{short_term_memory}

Reflect on your peers' opinions and update your stance. Provide your internal reasoning, compose a new tweet expressing your current opinion, and rate your updated belief on a 7-point Likert scale (1 = Strong Disbelief, 7 = Strong Belief).

Output exactly in this JSON format, and nothing else:
{{
"belief_score": <integer 1-7>,
"opinion": "<tweet>",
"reasoning": "<first-person internal monologue>"
}}
"""

daily_ltm_compress_prompt = """Review your existing long-term memory, today's short-term peer observations, and your new internal reasoning. Integrate these into a single, updated narrative. Crucially, compress the final text to be no longer than {max_words} words, maintaining continuity, retaining key attitude shifts, and discarding redundant details. Return only the compressed long-term memory text."""

intervention_cognitive_prompt = """You are simulating a real person with the following persona:
{persona_prompt}

Your overarching mindset is shaped by your long-term memory:
{long_term_memory}

Yesterday, your public stance was:
{previous_day_tweet}

Today, you observed the following peer opinions:
{short_term_memory}

BREAKING NEWS: The platform has just broadcasted the following official statement:
{refutation_text}

Reflect on how this official statement, combined with peer opinions, affects your view. Provide your internal reasoning, compose a new tweet, and output your updated belief on a 7-point Likert scale (1 = Strong Disbelief, 7 = Strong Belief).

Output exactly in this JSON format, and nothing else:
{{
"belief_score": <integer 1-7>,
"opinion": "<tweet>",
"reasoning": "<first-person internal monologue>"
}}
"""

intervention_ltm_compress_prompt = """Review your existing long-term memory, today's short-term peer observations, the breaking news refutation, and your new internal reasoning. Integrate these inputs into a single, coherent narrative that reflects your current worldview. Crucially, compress the final text to be no longer than {max_words} words, maintaining continuity, retaining key attitude shifts, and discarding redundant details. Return only the compressed long-term memory text."""

# Legacy aliases used by older call sites during transition
day0_broadcast_initial_prompt = day0_cognitive_prompt
reflecting_prompt = daily_stm_prompt
long_memory_prompt = daily_ltm_compress_prompt
update_opinion_prompt = daily_cognitive_prompt
