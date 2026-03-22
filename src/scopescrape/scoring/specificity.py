"""Specificity scoring using named entity count and text length.

Posts that name specific tools, companies, or products are more
actionable than vague complaints. Longer posts also tend to be
more detailed and specific.

Score range: 0.0 to 10.0

Note: spaCy NER is optional. If the en_core_web_sm model isn't
installed, we fall back to a regex-based entity approximation.
"""

from __future__ import annotations

import re
from typing import Optional

from scopescrape.log import get_logger
from scopescrape.models import RawPost

logger = get_logger(__name__)

# Fallback: common tool/product name patterns
# Matches capitalized words that look like product names
PRODUCT_PATTERN = re.compile(
    r"\b(?:[A-Z][a-z]+(?:[A-Z][a-z]+)+|"  # CamelCase: ClickUp, GitHub
    r"[A-Z]{2,}(?:\.?[a-z]+)?|"            # ACRONYMS: AWS, APIs
    r"[A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b"    # Title Case pairs: Google Docs
)

# 200+ common English words that often appear capitalized at sentence starts.
# Filter these out to reduce false positives.
CAPITALIZED_STOPWORDS = {
    # Pronouns
    "I", "You", "He", "She", "It", "We", "They", "Me", "Him", "Her", "Us", "Them",
    # Demonstratives/Articles
    "The", "This", "That", "These", "Those", "A", "An",
    # Question words
    "What", "When", "Where", "Which", "Why", "Who", "How", "Whom", "Whose",
    # Verbs (common sentence starters)
    "Is", "Are", "Was", "Were", "Be", "Been", "Being", "Have", "Has", "Had", "Do",
    "Does", "Did", "Will", "Would", "Should", "Could", "May", "Might", "Must", "Can",
    "Shall", "Ought", "Need", "Want", "Make", "Get", "Go", "Come", "See", "Think",
    "Know", "Find", "Try", "Use", "Work", "Call", "Ask", "Say", "Give", "Tell",
    # Conjunctions
    "And", "But", "Or", "Nor", "Yet", "So", "For", "As", "If", "Because",
    # Prepositions
    "In", "On", "At", "To", "From", "With", "Without", "By", "About", "Of", "Into",
    "Through", "During", "Before", "After", "Above", "Below", "Between", "Under",
    "Over", "Out", "Up", "Down", "Off", "Across", "Along", "Around", "Past",
    # Adverbs
    "Not", "Also", "Just", "Only", "Very", "Much", "More", "Most", "Less", "So",
    "Too", "Well", "Really", "Actually", "Already", "Still", "Even", "Here", "There",
    "Then", "Now", "Never", "Always", "Maybe", "Perhaps", "Probably", "Definitely",
    "Unfortunately", "Fortunately", "Obviously", "Clearly", "Actually", "Basically",
    # Common adjectives that start sentences
    "Good", "Bad", "Great", "Terrible", "Happy", "Sad", "Angry", "Calm", "Cold",
    "Hot", "Big", "Small", "Long", "Short", "Old", "New", "High", "Low", "Fast",
    "Slow", "Easy", "Hard", "Light", "Dark", "Bright", "Curious", "Obvious",
    "Sure", "True", "False", "Right", "Wrong", "Best", "Worst", "Better", "Worse",
    "Beautiful", "Ugly", "Clean", "Dirty", "Loud", "Quiet", "Soft", "Rough",
    # Built/similar words that can be sentence starters
    "Built", "Made", "Done", "Finished", "Started", "Ended", "Created", "Designed",
    "Developed", "Implemented", "Deployed", "Released", "Launched", "Shipped",
    # More common verbs that appear at start of sentences
    "Let", "Help", "Allow", "Prevent", "Stop", "Start", "Begin", "Continue",
    "Suggest", "Require", "Enable", "Disable", "Add", "Remove", "Change", "Update",
    # More articles and determiners
    "No", "Some", "Any", "All", "Each", "Every", "Either", "Neither", "Many",
    "Few", "Several", "Another", "Other", "Same", "Different", "Such",
    # More common nouns that might appear capitalized
    "One", "Two", "Three", "First", "Second", "Third", "Last", "Next", "Last",
    "Time", "Day", "Week", "Month", "Year", "Now", "Today", "Tomorrow", "Yesterday",
    # Proper nouns commonly used as sentence starters
    "Please", "Thank", "Thanks", "Thanks", "Hello", "Hi", "Yes", "No", "Okay", "Ok",
    "Sure", "Really", "Wow", "Oh", "Ah", "Well", "Anyway", "However", "Therefore",
    "Thus", "Hence", "Meanwhile", "While", "Whereas", "Although", "Though", "Even",
    "Despite", "Concerning", "Regarding", "Considering", "Generally", "Specifically",
    "Basically", "Essentially", "Literally", "Figuratively", "Apparently",
    # More helping verbs
    "Being", "Having", "Getting", "Making", "Taking", "Giving", "Putting", "Keeping",
    "Going", "Coming", "Doing", "Showing", "Leaving", "Following", "Including",
}

# Known SaaS/tech product names - boost list for better entity detection
KNOWN_SAAS_PRODUCTS = {
    "GitHub", "GitLab", "Bitbucket", "Slack", "Discord", "Teams", "Asana",
    "Monday", "ClickUp", "Notion", "Confluence", "Jira", "Linear", "Figma",
    "Sketch", "Framer", "Webflow", "Vercel", "Netlify", "Heroku", "AWS",
    "Azure", "GCP", "Google", "Salesforce", "HubSpot", "Pipedrive", "Zendesk",
    "Intercom", "Segment", "Mixpanel", "Amplitude", "Datadog", "New Relic",
    "Splunk", "Sumo", "Elastic", "MongoDB", "PostgreSQL", "MySQL", "Firebase",
    "Supabase", "PlanetScale", "Stripe", "Square", "PayPal", "Twilio", "SendGrid",
    "Mailchimp", "HubSpot", "Klaviyo", "Shopify", "WooCommerce", "Magento",
    "Dropbox", "OneDrive", "Google Drive", "Box", "Zoom", "Google Meet",
    "Miro", "Excalidraw", "InVision", "Zeplin", "Storybook", "Jest", "React",
    "Vue", "Angular", "Next.js", "Nuxt", "Svelte", "SvelteKit", "Remix",
    "Astro", "Deno", "Bun", "Node.js", "Python", "Ruby", "Java", "Go",
    "Rust", "C#", "Swift", "Kotlin", "TypeScript", "JavaScript", "PHP",
    "Laravel", "Django", "Rails", "Flask", "FastAPI", "Express", "Nest.js",
}



class SpecificityScorer:
    """Score posts by specificity of content (entities + length)."""

    def __init__(self, config: dict):
        self.config = config
        self._nlp = self._load_spacy()

    def _load_spacy(self):
        """Try to load spaCy model, return None if unavailable."""
        try:
            import spacy
            return spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            logger.info(
                "spaCy en_core_web_sm not found. Using regex fallback for entity extraction. "
                "Install with: python -m spacy download en_core_web_sm"
            )
            return None

    def score(self, post: RawPost) -> float:
        """Calculate specificity from entity count and text length.

        Args:
            post: The post to score.

        Returns:
            Score between 0.0 and 10.0.
        """
        text = post.full_text
        if not text:
            return 0.0

        # Component 1: Entity count (60% weight)
        entities = self.extract_entities(text)
        # 0 entities = 0, 1-2 = moderate, 3+ = high
        entity_score = min(10.0, len(entities) * 2.5)

        # Component 2: Text length (40% weight)
        # Short posts (<50 chars) = low, medium (50-500) = moderate, long (500+) = high
        length = len(text)
        if length < 50:
            length_score = 2.0
        elif length < 200:
            length_score = 5.0
        elif length < 500:
            length_score = 7.0
        else:
            length_score = min(10.0, 7.0 + (length - 500) / 500)

        combined = (entity_score * 0.6) + (length_score * 0.4)
        return round(min(10.0, max(0.0, combined)), 3)

    def extract_entities(self, text: str) -> list[str]:
        """Extract named entities (products, organizations, tools).

        Uses spaCy if available, falls back to regex pattern matching.

        Returns:
            Deduplicated list of entity strings.
        """
        if not text:
            return []

        if self._nlp is not None:
            return self._extract_spacy(text)
        return self._extract_regex(text)

    def _extract_spacy(self, text: str) -> list[str]:
        """Extract entities using spaCy NER."""
        doc = self._nlp(text[:5000])  # Cap length for performance
        entities = set()
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART", "FAC"):
                entities.add(ent.text.strip())
        return sorted(entities)

    def _extract_regex(self, text: str) -> list[str]:
        """Fallback entity extraction using regex patterns.

        Filters out:
        - Common English words that appear capitalized at sentence starts
        - Single common English words (too generic)
        - Boosts known SaaS/tech product names
        """
        matches = PRODUCT_PATTERN.findall(text)
        entities = set()

        for match in matches:
            # Filter 1: Reject if in stopwords (200+ common English words)
            if match in CAPITALIZED_STOPWORDS:
                continue

            # Filter 2: Minimum length 2 characters
            if len(match) < 2:
                continue

            # Filter 3: Reject single common English words (case-insensitive check)
            # This catches words like "Hello", "Today", "Really" that might pass through
            if len(match.split()) == 1 and match.lower() in {w.lower() for w in CAPITALIZED_STOPWORDS}:
                continue

            # Accept the match
            entities.add(match)

        # Return sorted list of entities
        return sorted(entities)
