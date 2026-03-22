"""ICP-to-subreddit recommendation engine for ScopeScrape.

This module provides a local mapping system that translates an Ideal Customer
Profile (ICP) description into recommended subreddits, keywords, and platforms
without requiring any external API calls or LLMs.

Core components:
  - Subreddit Taxonomy: 180+ curated subreddits organized by industry/niche
  - Keyword Generation: Pain-point-oriented keyword suggestions with stopword filtering
  - Platform Recommender: Smart cross-platform recommendations (Reddit, HN, GitHub, etc.)
  - Relevance Scoring: Improved matching with concept expansion and minimum thresholds
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# Comprehensive stopword set (150+ words)
STOPWORDS = {
    # Articles & determiners
    "a", "an", "the",
    # Prepositions
    "about", "above", "across", "after", "against", "along", "among", "around", "at",
    "before", "behind", "below", "beneath", "beside", "between", "beyond", "by",
    "down", "during", "except", "for", "from", "in", "inside", "into", "like",
    "near", "of", "off", "on", "out", "outside", "over", "since", "through", "to",
    "toward", "towards", "under", "underneath", "until", "up", "upon", "with", "within", "without",
    # Conjunctions
    "and", "but", "or", "nor", "yet", "so", "because", "if", "unless", "as",
    # Pronouns
    "i", "me", "he", "him", "she", "her", "it", "we", "us", "they", "them",
    "you", "what", "which", "who", "that", "this", "these", "those",
    # Verbs (common auxiliary/modal)
    "is", "are", "am", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "should", "could", "can", "may", "might", "must",
    # Other common words
    "very", "just", "only", "some", "any", "all", "each", "every", "both", "few", "many",
    "much", "more", "most", "no", "not", "own", "same", "such", "than", "too", "where", "when",
    "why", "how", "all", "each", "every", "both", "either", "neither", "one", "ones",
    "there", "here", "where", "when", "while", "as", "at", "by", "in", "to", "from",
    "as", "if", "also", "another", "any", "each", "every", "much", "other", "such", "well",
}

# Concept expansion: abbreviations and jargon mapped to fuller meanings
CONCEPT_SYNONYMS = {
    "0-to-1": ["early stage", "pre-pmf", "idea validation", "startup validation", "finding demand"],
    "0 to 1": ["early stage", "pre-pmf", "idea validation", "startup validation", "finding demand"],
    "crm": ["customer relationship management", "sales", "customer data"],
    "dtc": ["direct to consumer", "d2c", "direct sales", "ecommerce"],
    "b2b": ["business to business", "enterprise", "saas", "b2b sales"],
    "b2c": ["business to consumer", "consumer", "retail", "ecommerce"],
    "ai": ["artificial intelligence", "machine learning", "ml", "nlp", "deep learning"],
    "ml": ["machine learning", "artificial intelligence", "ai", "algorithms"],
    "api": ["application programming interface", "integration", "webhooks"],
    "mvp": ["minimum viable product", "prototype", "early launch"],
    "pmf": ["product market fit", "product-market fit", "market validation"],
    "ux": ["user experience", "usability", "interface design"],
    "ui": ["user interface", "design", "frontend"],
    "seo": ["search engine optimization", "organic search", "ranking"],
    "ppc": ["pay per click", "google ads", "advertising"],
    "devops": ["development operations", "infrastructure", "deployment"],
    "ci/cd": ["continuous integration", "continuous deployment", "automation"],
    "daas": ["data as a service", "data platform"],
    "paas": ["platform as a service", "cloud platform"],
    "iaas": ["infrastructure as a service", "cloud infrastructure"],
    "llm": ["large language model", "language model", "gpt"],
    "iot": ["internet of things", "connected devices"],
    "vpn": ["virtual private network", "privacy", "security"],
    "scrapper": ["scraper", "scraping", "web scraping", "data extraction"],
    "demand capture": ["market research", "pain point discovery", "finding demand", "demand validation"],
}


@dataclass
class Subreddit:
    """Represents a subreddit with metadata for recommendation."""

    name: str
    category: str
    typical_audience: str
    relevance_keywords: list[str]
    platform: str = "reddit"

    def score_relevance(self, icp_keywords: list[str]) -> float:
        """
        Score this subreddit's relevance to an ICP based on keyword overlap.
        Requires at least 2 matches for relevance > 0.
        """
        if not icp_keywords:
            return 0.0

        matches = sum(1 for keyword in icp_keywords if keyword.lower() in [k.lower() for k in self.relevance_keywords])

        # Require at least 2 matches
        if matches < 2:
            return 0.0

        return min(matches / len(self.relevance_keywords), 1.0)


@dataclass
class RecommendationResult:
    """Result of a recommendation query."""

    subreddits: list[dict]
    keywords: list[str]
    platforms: list[str]
    icp_summary: str


class Recommender:
    """ICP-to-subreddit recommendation engine.

    Provides smart subreddit, keyword, and platform recommendations for any ICP
    described in plain text. Uses no external APIs or ML—purely local mappings.
    """

    # Competitor mapping for common niches
    COMPETITORS = {
        "crm": ["Salesforce", "HubSpot", "Pipedrive", "Zoho", "Dynamics", "Insightly", "Freshsales"],
        "email marketing": ["Mailchimp", "ConvertKit", "ActiveCampaign", "GetResponse", "Klaviyo", "Brevo"],
        "project management": ["Asana", "Monday", "ClickUp", "Notion", "Trello", "Jira", "Linear"],
        "marketing automation": ["HubSpot", "Marketo", "Pardot", "ActiveCampaign", "ConvertKit", "Braze"],
        "accounting": ["QuickBooks", "FreshBooks", "Wave", "Xero", "NetSuite", "Zoho Books"],
        "ecommerce": ["Shopify", "WooCommerce", "Magento", "BigCommerce", "Wix", "Square Online"],
        "saas": ["Stripe", "Chargebee", "Recurly", "ProfitWell", "Zuora", "Paddle"],
        "analytics": ["Google Analytics", "Mixpanel", "Amplitude", "Segment", "Heap", "Plausible"],
        "learning": ["Udemy", "Coursera", "LinkedIn Learning", "Teachable", "Kajabi", "Thinkific"],
        "video conferencing": ["Zoom", "Teams", "Google Meet", "WebEx", "Slack", "Whereby"],
        "design": ["Figma", "Adobe XD", "Sketch", "InVision", "Framer", "Penpot"],
        "documentation": ["Notion", "Confluence", "GitBook", "Slite", "Documize", "ReadTheDocs"],
        "vpn": ["NordVPN", "ExpressVPN", "Surfshark", "ProtonVPN", "CyberGhost", "Mullvad"],
        "password manager": ["1Password", "LastPass", "Dashlane", "Bitwarden", "KeePass"],
        "monitoring": ["Datadog", "New Relic", "Prometheus", "Grafana", "Sentry", "Honeycomb"],
        "scraping": ["Apify", "Scrapy", "BeautifulSoup", "Selenium", "Puppeteer"],
        "payments": ["Stripe", "Square", "PayPal", "Braintree", "Adyen"],
        "webhosting": ["AWS", "DigitalOcean", "Linode", "Netlify", "Vercel", "Heroku"],
    }

    def __init__(self):
        """Initialize the recommender with the subreddit taxonomy."""
        self.subreddits = self._build_subreddit_taxonomy()

    def _build_subreddit_taxonomy(self) -> list[Subreddit]:
        """Build the curated subreddit taxonomy (180+ subreddits)."""
        return [
            # SaaS / Software
            Subreddit(
                name="r/saas",
                category="SaaS",
                typical_audience="SaaS founders, entrepreneurs, and business owners",
                relevance_keywords=["saas", "software as a service", "subscription", "recurring revenue", "subscription model"],
            ),
            Subreddit(
                name="r/startups",
                category="SaaS",
                typical_audience="Startup founders and early-stage entrepreneurs",
                relevance_keywords=["startup", "founding", "launch", "funding", "growth stage", "early stage", "idea validation", "pre-pmf", "finding demand"],
            ),
            Subreddit(
                name="r/Entrepreneur",
                category="SaaS",
                typical_audience="Business entrepreneurs at all stages",
                relevance_keywords=["entrepreneur", "business", "venture", "idea validation", "revenue generation", "startup validation", "finding demand"],
            ),
            Subreddit(
                name="r/smallbusiness",
                category="SaaS",
                typical_audience="Small business owners and operators",
                relevance_keywords=["small business", "smb", "local business", "owner operator"],
            ),
            Subreddit(
                name="r/indiehackers",
                category="SaaS",
                typical_audience="Independent developers and makers",
                relevance_keywords=["indie", "maker", "solo founder", "bootstrapped", "side project", "finding demand", "idea validation", "pre-pmf"],
            ),
            Subreddit(
                name="r/microsaas",
                category="SaaS",
                typical_audience="Micro-SaaS founders (typically <$2M ARR)",
                relevance_keywords=["microsaas", "micro saas", "niche", "solopreneur", "product launch"],
            ),
            Subreddit(
                name="r/SaaS_Sales",
                category="SaaS",
                typical_audience="SaaS sales professionals and revenue leaders",
                relevance_keywords=["sales", "revenue", "customer acquisition", "outbound", "pipeline"],
            ),
            Subreddit(
                name="r/GrowMyBusiness",
                category="SaaS",
                typical_audience="Business growth strategists and marketers",
                relevance_keywords=["growth", "scaling", "expansion", "marketing", "retention"],
            ),
            # Developer Tools / Programming
            Subreddit(
                name="r/programming",
                category="Developer Tools",
                typical_audience="Software developers and computer scientists",
                relevance_keywords=["programming", "code", "developer", "software development", "api"],
            ),
            Subreddit(
                name="r/webdev",
                category="Developer Tools",
                typical_audience="Web developers and full-stack engineers",
                relevance_keywords=["web development", "frontend", "backend", "fullstack", "html"],
            ),
            Subreddit(
                name="r/devops",
                category="Developer Tools",
                typical_audience="DevOps engineers and infrastructure specialists",
                relevance_keywords=["devops", "infrastructure", "deployment", "cicd", "operations"],
            ),
            Subreddit(
                name="r/learnprogramming",
                category="Developer Tools",
                typical_audience="Programming students and self-taught developers",
                relevance_keywords=["learn", "tutorial", "course", "beginner", "education"],
            ),
            Subreddit(
                name="r/reactjs",
                category="Developer Tools",
                typical_audience="React.js developers",
                relevance_keywords=["react", "javascript", "frontend framework", "jsx", "component"],
            ),
            Subreddit(
                name="r/nextjs",
                category="Developer Tools",
                typical_audience="Next.js developers",
                relevance_keywords=["nextjs", "next.js", "react framework", "fullstack", "ssr"],
            ),
            Subreddit(
                name="r/vuejs",
                category="Developer Tools",
                typical_audience="Vue.js developers",
                relevance_keywords=["vuejs", "vue", "javascript framework", "frontend"],
            ),
            Subreddit(
                name="r/angular",
                category="Developer Tools",
                typical_audience="Angular developers",
                relevance_keywords=["angular", "typescript", "frontend framework", "spa"],
            ),
            Subreddit(
                name="r/python",
                category="Developer Tools",
                typical_audience="Python developers",
                relevance_keywords=["python", "django", "flask", "data science", "automation"],
            ),
            Subreddit(
                name="r/golang",
                category="Developer Tools",
                typical_audience="Go/Golang developers",
                relevance_keywords=["golang", "go language", "backend", "concurrent", "performance"],
            ),
            Subreddit(
                name="r/rust",
                category="Developer Tools",
                typical_audience="Rust developers",
                relevance_keywords=["rust", "systems programming", "memory safety", "performance"],
            ),
            Subreddit(
                name="r/node",
                category="Developer Tools",
                typical_audience="Node.js developers",
                relevance_keywords=["node", "nodejs", "javascript backend", "express"],
            ),
            Subreddit(
                name="r/docker",
                category="Developer Tools",
                typical_audience="Docker and containerization specialists",
                relevance_keywords=["docker", "containers", "containerization", "deployment"],
            ),
            Subreddit(
                name="r/kubernetes",
                category="Developer Tools",
                typical_audience="Kubernetes and orchestration specialists",
                relevance_keywords=["kubernetes", "k8s", "container orchestration", "devops"],
            ),
            Subreddit(
                name="r/githubcopilot",
                category="Developer Tools",
                typical_audience="GitHub Copilot users and AI coding enthusiasts",
                relevance_keywords=["copilot", "ai coding", "code generation", "productivity"],
            ),
            Subreddit(
                name="r/webhosting",
                category="Developer Tools",
                typical_audience="Web hosting and server management",
                relevance_keywords=["webhosting", "hosting", "server", "vps", "cloud"],
            ),
            Subreddit(
                name="r/selfhosted",
                category="Developer Tools",
                typical_audience="Self-hosted software and open source enthusiasts",
                relevance_keywords=["selfhosted", "self hosted", "open source", "infrastructure"],
            ),
            Subreddit(
                name="r/homelab",
                category="Developer Tools",
                typical_audience="Home lab builders and enthusiasts",
                relevance_keywords=["homelab", "home lab", "server", "networking", "infrastructure"],
            ),
            Subreddit(
                name="r/webscraping",
                category="Developer Tools",
                typical_audience="Web scraping practitioners and data collectors",
                relevance_keywords=["webscraping", "web scraping", "scraper", "scraping", "data collection", "crawling", "demand capture"],
            ),
            # AI / ML
            Subreddit(
                name="r/artificial",
                category="AI/ML",
                typical_audience="Artificial intelligence practitioners and researchers",
                relevance_keywords=["artificial intelligence", "ai", "machine learning", "nlp"],
            ),
            Subreddit(
                name="r/MachineLearning",
                category="AI/ML",
                typical_audience="Machine learning engineers and data scientists",
                relevance_keywords=["machine learning", "ml", "deep learning", "neural networks"],
            ),
            Subreddit(
                name="r/LocalLLaMA",
                category="AI/ML",
                typical_audience="Local LLM and open-source AI enthusiasts",
                relevance_keywords=["llama", "local llm", "open source ai", "self hosted"],
            ),
            Subreddit(
                name="r/ChatGPT",
                category="AI/ML",
                typical_audience="ChatGPT users and prompt engineers",
                relevance_keywords=["chatgpt", "gpt", "prompt engineering", "openai"],
            ),
            Subreddit(
                name="r/midjourney",
                category="AI/ML",
                typical_audience="Midjourney and AI image generation users",
                relevance_keywords=["midjourney", "image generation", "ai art", "generative"],
            ),
            Subreddit(
                name="r/StableDiffusion",
                category="AI/ML",
                typical_audience="Stable Diffusion and open-source image AI users",
                relevance_keywords=["stable diffusion", "image generation", "open source ai"],
            ),
            # Marketing
            Subreddit(
                name="r/marketing",
                category="Marketing",
                typical_audience="Marketing professionals and strategists",
                relevance_keywords=["marketing", "strategy", "campaigns", "brand"],
            ),
            Subreddit(
                name="r/digital_marketing",
                category="Marketing",
                typical_audience="Digital marketers and online marketing specialists",
                relevance_keywords=["digital marketing", "online marketing", "campaigns", "growth"],
            ),
            Subreddit(
                name="r/SEO",
                category="Marketing",
                typical_audience="SEO specialists and search marketers",
                relevance_keywords=["seo", "search engine optimization", "organic search", "ranking"],
            ),
            Subreddit(
                name="r/socialmedia",
                category="Marketing",
                typical_audience="Social media managers and content creators",
                relevance_keywords=["social media", "twitter", "instagram", "content"],
            ),
            Subreddit(
                name="r/twitter_marketing",
                category="Marketing",
                typical_audience="Twitter marketing and engagement specialists",
                relevance_keywords=["twitter", "x", "social media marketing", "engagement"],
            ),
            Subreddit(
                name="r/WhatsApp",
                category="Marketing",
                typical_audience="WhatsApp messaging and marketing professionals",
                relevance_keywords=["whatsapp", "messaging", "marketing", "communication"],
            ),
            Subreddit(
                name="r/PPC",
                category="Marketing",
                typical_audience="Pay-per-click and SEM specialists",
                relevance_keywords=["ppc", "google ads", "facebook ads", "advertising"],
            ),
            Subreddit(
                name="r/content_marketing",
                category="Marketing",
                typical_audience="Content marketers and copywriters",
                relevance_keywords=["content marketing", "copywriting", "blog", "strategy"],
            ),
            Subreddit(
                name="r/emailmarketing",
                category="Marketing",
                typical_audience="Email marketing professionals",
                relevance_keywords=["email marketing", "email automation", "newsletter", "outreach"],
            ),
            Subreddit(
                name="r/copywriting",
                category="Marketing",
                typical_audience="Copywriters and persuasion specialists",
                relevance_keywords=["copywriting", "persuasion", "sales", "messaging"],
            ),
            Subreddit(
                name="r/ProductMarketing",
                category="Marketing",
                typical_audience="Product marketers and go-to-market strategists",
                relevance_keywords=["product marketing", "gtm", "positioning", "messaging"],
            ),
            Subreddit(
                name="r/socialmediamarketing",
                category="Marketing",
                typical_audience="Social media marketing specialists",
                relevance_keywords=["social media marketing", "social marketing", "engagement", "growth"],
            ),
            # E-commerce
            Subreddit(
                name="r/ecommerce",
                category="E-commerce",
                typical_audience="E-commerce business owners and operators",
                relevance_keywords=["ecommerce", "online store", "retail", "sales"],
            ),
            Subreddit(
                name="r/shopify",
                category="E-commerce",
                typical_audience="Shopify store owners and developers",
                relevance_keywords=["shopify", "store", "dropshipping", "sales"],
            ),
            Subreddit(
                name="r/dropship",
                category="E-commerce",
                typical_audience="Dropshipping business owners",
                relevance_keywords=["dropshipping", "supplier", "products", "fulfillment"],
            ),
            Subreddit(
                name="r/FulfillmentByAmazon",
                category="E-commerce",
                typical_audience="Amazon FBA sellers",
                relevance_keywords=["fba", "amazon fulfillment", "fulfillment", "seller"],
            ),
            Subreddit(
                name="r/Etsy",
                category="E-commerce",
                typical_audience="Etsy sellers and handmade business owners",
                relevance_keywords=["etsy", "handmade", "crafts", "marketplace"],
            ),
            Subreddit(
                name="r/AmazonSeller",
                category="E-commerce",
                typical_audience="Amazon marketplace sellers",
                relevance_keywords=["amazon seller", "seller", "fba", "ppc"],
            ),
            Subreddit(
                name="r/FBAOnlineArbitrage",
                category="E-commerce",
                typical_audience="FBA and online arbitrage sellers",
                relevance_keywords=["fba", "arbitrage", "online arbitrage", "reselling"],
            ),
            Subreddit(
                name="r/AmazonMerch",
                category="E-commerce",
                typical_audience="Amazon Merch on Demand creators",
                relevance_keywords=["amazon merch", "merch creator", "tshirt design", "apparel", "merchandise"],
            ),
            # Finance / Fintech
            Subreddit(
                name="r/fintech",
                category="Finance",
                typical_audience="Fintech entrepreneurs and professionals",
                relevance_keywords=["fintech", "finance technology", "payments", "blockchain"],
            ),
            Subreddit(
                name="r/personalfinance",
                category="Finance",
                typical_audience="Personal finance enthusiasts and budgeters",
                relevance_keywords=["personal finance", "savings", "budgeting", "investment", "mortgage", "homebuyers", "credit score"],
            ),
            Subreddit(
                name="r/CreditCards",
                category="Finance",
                typical_audience="Credit card users and rewards enthusiasts",
                relevance_keywords=["credit cards", "rewards", "points", "benefits"],
            ),
            Subreddit(
                name="r/investing",
                category="Finance",
                typical_audience="Stock and investment enthusiasts",
                relevance_keywords=["investing", "stocks", "portfolio", "returns"],
            ),
            Subreddit(
                name="r/accounting",
                category="Finance",
                typical_audience="Accountants and bookkeepers",
                relevance_keywords=["accounting", "bookkeeping", "taxes", "finance"],
            ),
            Subreddit(
                name="r/forex",
                category="Finance",
                typical_audience="Foreign exchange traders",
                relevance_keywords=["forex", "trading", "currency", "markets"],
            ),
            Subreddit(
                name="r/CryptoCurrency",
                category="Finance",
                typical_audience="Cryptocurrency investors and traders",
                relevance_keywords=["crypto", "bitcoin", "ethereum", "blockchain"],
            ),
            Subreddit(
                name="r/payments",
                category="Finance",
                typical_audience="Payment systems and processing professionals",
                relevance_keywords=["payments", "payment processing", "stripe", "transaction"],
            ),
            Subreddit(
                name="r/stripe",
                category="Finance",
                typical_audience="Stripe users and payment processing",
                relevance_keywords=["stripe", "payment processing", "payments", "checkout"],
            ),
            # Healthcare
            Subreddit(
                name="r/healthIT",
                category="Healthcare",
                typical_audience="Health IT professionals and healthcare technologists",
                relevance_keywords=["health it", "healthcare technology", "ehr", "telemedicine"],
            ),
            Subreddit(
                name="r/medicine",
                category="Healthcare",
                typical_audience="Medical professionals and physicians",
                relevance_keywords=["medicine", "doctor", "diagnosis", "treatment"],
            ),
            Subreddit(
                name="r/nursing",
                category="Healthcare",
                typical_audience="Nurses and nursing professionals",
                relevance_keywords=["nursing", "nurse", "patient care", "rn"],
            ),
            Subreddit(
                name="r/dentistry",
                category="Healthcare",
                typical_audience="Dentists and dental professionals",
                relevance_keywords=["dentistry", "dental", "teeth", "orthodontics"],
            ),
            Subreddit(
                name="r/pharmacy",
                category="Healthcare",
                typical_audience="Pharmacists and pharmacy technicians",
                relevance_keywords=["pharmacy", "pharmacist", "medications", "pharmaceuticals"],
            ),
            Subreddit(
                name="r/Fitness",
                category="Healthcare",
                typical_audience="Fitness enthusiasts and trainers",
                relevance_keywords=["fitness", "exercise", "gym", "training"],
            ),
            Subreddit(
                name="r/petcare",
                category="Healthcare",
                typical_audience="Pet owners and animal care enthusiasts",
                relevance_keywords=["petcare", "pet care", "pet", "animal health"],
            ),
            Subreddit(
                name="r/veterinary",
                category="Healthcare",
                typical_audience="Veterinarians and veterinary professionals",
                relevance_keywords=["veterinary", "vet", "animal health", "pets"],
            ),
            # Real Estate
            Subreddit(
                name="r/realestate",
                category="Real Estate",
                typical_audience="Real estate professionals and enthusiasts",
                relevance_keywords=["real estate", "property", "homes", "agent"],
            ),
            Subreddit(
                name="r/RealEstateInvesting",
                category="Real Estate",
                typical_audience="Real estate investors",
                relevance_keywords=["real estate investing", "rental property", "roi"],
            ),
            Subreddit(
                name="r/CommercialRealEstate",
                category="Real Estate",
                typical_audience="Commercial real estate professionals",
                relevance_keywords=["commercial real estate", "office", "retail", "lease"],
            ),
            Subreddit(
                name="r/PropertyManagement",
                category="Real Estate",
                typical_audience="Property managers and landlords",
                relevance_keywords=["property management", "tenant", "landlord", "maintenance"],
            ),
            Subreddit(
                name="r/Landlord",
                category="Real Estate",
                typical_audience="Landlords and property owners",
                relevance_keywords=["landlord", "tenant", "rental", "lease"],
            ),
            Subreddit(
                name="r/FirstTimeHomeBuyer",
                category="Real Estate",
                typical_audience="First-time home buyers seeking guidance",
                relevance_keywords=["first-time", "homebuyers", "mortgage", "home buying", "down payment", "closing costs"],
            ),
            Subreddit(
                name="r/HomeImprovement",
                category="Real Estate",
                typical_audience="Home improvement and renovation enthusiasts",
                relevance_keywords=["home improvement", "renovation", "diy", "construction"],
            ),
            Subreddit(
                name="r/Construction",
                category="Real Estate",
                typical_audience="Construction professionals and contractors",
                relevance_keywords=["construction", "contractor", "building", "project"],
            ),
            # Education
            Subreddit(
                name="r/edtech",
                category="Education",
                typical_audience="EdTech entrepreneurs and educators",
                relevance_keywords=["edtech", "education technology", "learning platform", "online course"],
            ),
            Subreddit(
                name="r/Teachers",
                category="Education",
                typical_audience="Teachers and educators",
                relevance_keywords=["teaching", "education", "classroom", "students"],
            ),
            Subreddit(
                name="r/OnlineEducation",
                category="Education",
                typical_audience="Online course creators and learners",
                relevance_keywords=["online education", "course", "learning", "training"],
            ),
            Subreddit(
                name="r/languagelearning",
                category="Education",
                typical_audience="Language learners",
                relevance_keywords=["language learning", "foreign language", "bilingual"],
            ),
            # Design
            Subreddit(
                name="r/design",
                category="Design",
                typical_audience="Designers and creative professionals",
                relevance_keywords=["design", "ux", "ui", "creative"],
            ),
            Subreddit(
                name="r/web_design",
                category="Design",
                typical_audience="Web designers and front-end designers",
                relevance_keywords=["web design", "website", "layout", "ux"],
            ),
            Subreddit(
                name="r/UI_Design",
                category="Design",
                typical_audience="UI/UX designers",
                relevance_keywords=["ui design", "user interface", "design system"],
            ),
            Subreddit(
                name="r/userexperience",
                category="Design",
                typical_audience="UX designers and user experience specialists",
                relevance_keywords=["user experience", "ux", "usability", "design"],
            ),
            Subreddit(
                name="r/graphic_design",
                category="Design",
                typical_audience="Graphic designers and visual artists",
                relevance_keywords=["graphic design", "visual design", "branding"],
            ),
            Subreddit(
                name="r/FigmaDesign",
                category="Design",
                typical_audience="Figma users and design tool users",
                relevance_keywords=["figma", "design tool", "prototyping", "ui"],
            ),
            # Legal / LegalTech
            Subreddit(
                name="r/legaltech",
                category="Legal",
                typical_audience="Legal tech entrepreneurs and professionals",
                relevance_keywords=["legaltech", "legal technology", "law", "automation"],
            ),
            Subreddit(
                name="r/law",
                category="Legal",
                typical_audience="Lawyers and legal professionals",
                relevance_keywords=["law", "legal", "attorney", "lawyer"],
            ),
            Subreddit(
                name="r/lawyers",
                category="Legal",
                typical_audience="Lawyers and law students",
                relevance_keywords=["lawyers", "legal profession", "bar exam"],
            ),
            Subreddit(
                name="r/legaladvice",
                category="Legal",
                typical_audience="Legal advice seekers",
                relevance_keywords=["legal advice", "legal help", "law", "attorney"],
            ),
            Subreddit(
                name="r/lawfirm",
                category="Legal",
                typical_audience="Law firm owners and managers",
                relevance_keywords=["law firm", "legal practice", "attorney", "firm"],
            ),
            # HR / Recruiting
            Subreddit(
                name="r/humanresources",
                category="HR/Recruiting",
                typical_audience="HR professionals and managers",
                relevance_keywords=["hr", "human resources", "hiring", "employees"],
            ),
            Subreddit(
                name="r/recruiting",
                category="HR/Recruiting",
                typical_audience="Recruiters and talent professionals",
                relevance_keywords=["recruiting", "recruitment", "hiring", "talent"],
            ),
            Subreddit(
                name="r/remotejobs",
                category="HR/Recruiting",
                typical_audience="Remote work seekers and employers",
                relevance_keywords=["remote work", "work from home", "distributed"],
            ),
            Subreddit(
                name="r/jobs",
                category="HR/Recruiting",
                typical_audience="Job seekers and employers",
                relevance_keywords=["jobs", "employment", "career", "hiring"],
            ),
            # Productivity / Tools
            Subreddit(
                name="r/productivity",
                category="Productivity",
                typical_audience="Productivity enthusiasts and tool users",
                relevance_keywords=["productivity", "efficiency", "workflow", "organization"],
            ),
            Subreddit(
                name="r/Notion",
                category="Productivity",
                typical_audience="Notion users and workspace builders",
                relevance_keywords=["notion", "workspace", "database", "productivity"],
            ),
            Subreddit(
                name="r/ObsidianMD",
                category="Productivity",
                typical_audience="Obsidian note-taking enthusiasts",
                relevance_keywords=["obsidian", "note taking", "markdown", "knowledge"],
            ),
            Subreddit(
                name="r/nocode",
                category="Productivity",
                typical_audience="No-code and low-code platform users",
                relevance_keywords=["no code", "low code", "automation", "builder"],
            ),
            # Gaming
            Subreddit(
                name="r/gamedev",
                category="Gaming",
                typical_audience="Game developers and indie game creators",
                relevance_keywords=["game development", "gamedev", "unity", "unreal"],
            ),
            Subreddit(
                name="r/indiegaming",
                category="Gaming",
                typical_audience="Indie game developers and players",
                relevance_keywords=["indie games", "game dev", "steam", "itch"],
            ),
            Subreddit(
                name="r/gaming",
                category="Gaming",
                typical_audience="Gamers and gaming enthusiasts",
                relevance_keywords=["gaming", "games", "video games", "community"],
            ),
            # Crypto / Web3
            Subreddit(
                name="r/defi",
                category="Crypto/Web3",
                typical_audience="DeFi users and developers",
                relevance_keywords=["defi", "decentralized finance", "smart contracts"],
            ),
            Subreddit(
                name="r/web3",
                category="Crypto/Web3",
                typical_audience="Web3 and blockchain enthusiasts",
                relevance_keywords=["web3", "blockchain", "decentralized", "nft"],
            ),
            Subreddit(
                name="r/ethereum",
                category="Crypto/Web3",
                typical_audience="Ethereum developers and users",
                relevance_keywords=["ethereum", "eth", "smart contracts", "defi"],
            ),
            # Data & Analytics
            Subreddit(
                name="r/datascience",
                category="Data",
                typical_audience="Data scientists and analysts",
                relevance_keywords=["data science", "analytics", "data analysis", "ml"],
            ),
            Subreddit(
                name="r/dataengineering",
                category="Data",
                typical_audience="Data engineers and pipeline builders",
                relevance_keywords=["data engineering", "data pipeline", "etl", "database"],
            ),
            Subreddit(
                name="r/analytics",
                category="Data",
                typical_audience="Analytics professionals",
                relevance_keywords=["analytics", "metrics", "dashboards", "reporting"],
            ),
            Subreddit(
                name="r/datasets",
                category="Data",
                typical_audience="Data enthusiasts and researchers",
                relevance_keywords=["datasets", "data", "research", "open data"],
            ),
            # Transportation / Logistics
            Subreddit(
                name="r/Truckers",
                category="Transportation",
                typical_audience="Truckers and trucking professionals",
                relevance_keywords=["trucking", "trucker", "freight", "logistics"],
            ),
            Subreddit(
                name="r/fleet",
                category="Transportation",
                typical_audience="Fleet managers and operators",
                relevance_keywords=["fleet", "fleet management", "vehicles", "logistics"],
            ),
            # Services / Trades
            Subreddit(
                name="r/grooming",
                category="Services",
                typical_audience="Pet groomers and salon professionals",
                relevance_keywords=["grooming", "pet grooming", "salon", "styling"],
            ),
            Subreddit(
                name="r/petbusiness",
                category="Services",
                typical_audience="Pet business owners and operators",
                relevance_keywords=["pet business", "pet services", "pets", "salon"],
            ),
            Subreddit(
                name="r/weddingplanning",
                category="Services",
                typical_audience="Wedding planners and couples",
                relevance_keywords=["wedding planning", "wedding", "event planning", "bride"],
            ),
            Subreddit(
                name="r/churchops",
                category="Services",
                typical_audience="Church operations and leadership",
                relevance_keywords=["church", "church operations", "faith community", "ministry"],
            ),
            # Project & Product Development
            Subreddit(
                name="r/SideProject",
                category="Entrepreneurship",
                typical_audience="Side project builders and makers",
                relevance_keywords=["side project", "side hustle", "maker", "indie"],
            ),
            Subreddit(
                name="r/buildinpublic",
                category="Entrepreneurship",
                typical_audience="Builders sharing their journey publicly",
                relevance_keywords=["building public", "indie hacker", "transparency", "journey"],
            ),
            # Platform-specific
            Subreddit(
                name="r/stackoverflow",
                category="Developer Tools",
                typical_audience="Stack Overflow users and Q&A enthusiasts",
                relevance_keywords=["stackoverflow", "q&a", "questions", "answers"],
            ),
            # General Business
            Subreddit(
                name="r/business",
                category="General Business",
                typical_audience="Business professionals and leaders",
                relevance_keywords=["business", "management", "strategy", "operations"],
            ),
            Subreddit(
                name="r/freelance",
                category="General Business",
                typical_audience="Freelancers and consultants",
                relevance_keywords=["freelance", "consulting", "independent contractor"],
            ),
            Subreddit(
                name="r/consulting",
                category="General Business",
                typical_audience="Management consultants and advisors",
                relevance_keywords=["consulting", "advisory", "strategy", "client"],
            ),
            # Community specific
            Subreddit(
                name="r/IAmA",
                category="General",
                typical_audience="People interested in Q&A with interesting individuals",
                relevance_keywords=["ama", "ask me anything", "community", "discussion"],
            ),
            Subreddit(
                name="r/AskReddit",
                category="General",
                typical_audience="Discussion and community engagement",
                relevance_keywords=["ask reddit", "discussion", "community", "feedback"],
            ),
            # Additional Developer/Tech Communities
            Subreddit(
                name="r/javascript",
                category="Developer Tools",
                typical_audience="JavaScript developers and enthusiasts",
                relevance_keywords=["javascript", "js", "web development", "frontend"],
            ),
            Subreddit(
                name="r/typescript",
                category="Developer Tools",
                typical_audience="TypeScript developers",
                relevance_keywords=["typescript", "ts", "javascript", "static typing"],
            ),
            Subreddit(
                name="r/php",
                category="Developer Tools",
                typical_audience="PHP developers",
                relevance_keywords=["php", "web development", "backend", "laravel"],
            ),
            Subreddit(
                name="r/java",
                category="Developer Tools",
                typical_audience="Java developers",
                relevance_keywords=["java", "backend", "enterprise", "springboot"],
            ),
            Subreddit(
                name="r/csharp",
                category="Developer Tools",
                typical_audience="C# developers",
                relevance_keywords=["csharp", "c#", ".net", "backend"],
            ),
            Subreddit(
                name="r/cpp",
                category="Developer Tools",
                typical_audience="C++ developers",
                relevance_keywords=["cpp", "c++", "systems", "performance"],
            ),
            Subreddit(
                name="r/database",
                category="Developer Tools",
                typical_audience="Database architects and engineers",
                relevance_keywords=["database", "sql", "postgresql", "mongodb"],
            ),
            Subreddit(
                name="r/aws",
                category="Developer Tools",
                typical_audience="Amazon Web Services users",
                relevance_keywords=["aws", "amazon web services", "cloud", "infrastructure"],
            ),
            Subreddit(
                name="r/azure",
                category="Developer Tools",
                typical_audience="Microsoft Azure users",
                relevance_keywords=["azure", "microsoft", "cloud", "infrastructure"],
            ),
            Subreddit(
                name="r/gcp",
                category="Developer Tools",
                typical_audience="Google Cloud Platform users",
                relevance_keywords=["gcp", "google cloud", "cloud", "infrastructure"],
            ),
            # Additional Product/Growth Communities
            Subreddit(
                name="r/ProductManagement",
                category="SaaS",
                typical_audience="Product managers and strategists",
                relevance_keywords=["product management", "product strategy", "roadmap"],
            ),
            Subreddit(
                name="r/growthacking",
                category="Marketing",
                typical_audience="Growth hackers and experimentation enthusiasts",
                relevance_keywords=["growth hacking", "growth", "acquisition", "viral"],
            ),
            Subreddit(
                name="r/conversion",
                category="Marketing",
                typical_audience="Conversion rate optimization specialists",
                relevance_keywords=["conversion optimization", "cro", "funnel", "ab testing"],
            ),
            # Additional Sales Communities
            Subreddit(
                name="r/sales",
                category="SaaS",
                typical_audience="Sales professionals and managers",
                relevance_keywords=["sales", "selling", "b2b sales", "pipeline"],
            ),
            Subreddit(
                name="r/salesforce",
                category="SaaS",
                typical_audience="Salesforce users and admins",
                relevance_keywords=["salesforce", "crm", "sales cloud", "force"],
            ),
            # Additional Startup/Innovation
            Subreddit(
                name="r/ProductHunt",
                category="SaaS",
                typical_audience="Product Hunt users and makers",
                relevance_keywords=["product hunt", "launch", "product release", "maker"],
            ),
            Subreddit(
                name="r/Pitch",
                category="SaaS",
                typical_audience="Pitch deck and fundraising experts",
                relevance_keywords=["pitch deck", "fundraising", "venture", "presentation"],
            ),
            # Additional B2B/Enterprise
            Subreddit(
                name="r/enterprise",
                category="SaaS",
                typical_audience="Enterprise software and architecture",
                relevance_keywords=["enterprise", "b2b", "large scale", "infrastructure"],
            ),
            # Additional Data Communities
            Subreddit(
                name="r/bigdata",
                category="Data",
                typical_audience="Big data engineers and specialists",
                relevance_keywords=["big data", "hadoop", "spark", "distributed"],
            ),
            Subreddit(
                name="r/databases",
                category="Data",
                typical_audience="Database enthusiasts",
                relevance_keywords=["databases", "sql", "nosql", "query"],
            ),
            # Additional Marketing Communities
            Subreddit(
                name="r/emailfunnel",
                category="Marketing",
                typical_audience="Email funnel and sequence specialists",
                relevance_keywords=["email funnel", "email sequence", "automation"],
            ),
            Subreddit(
                name="r/affiliate",
                category="Marketing",
                typical_audience="Affiliate marketers",
                relevance_keywords=["affiliate marketing", "affiliate", "commission"],
            ),
            Subreddit(
                name="r/Adops",
                category="Marketing",
                typical_audience="Ad operations and media buying",
                relevance_keywords=["ad ops", "media buying", "advertising", "campaigns"],
            ),
            # Additional Customer Success
            Subreddit(
                name="r/CustomerSuccess",
                category="SaaS",
                typical_audience="Customer success and support professionals",
                relevance_keywords=["customer success", "support", "retention", "churn"],
            ),
            # Additional Design/Creative
            Subreddit(
                name="r/branddesign",
                category="Design",
                typical_audience="Brand designers and identity specialists",
                relevance_keywords=["brand design", "branding", "logo", "identity"],
            ),
            Subreddit(
                name="r/webdesign_critique",
                category="Design",
                typical_audience="Web design critics and feedback providers",
                relevance_keywords=["web design", "critique", "feedback", "portfolio"],
            ),
            # Additional E-commerce
            Subreddit(
                name="r/dropshippingDE",
                category="E-commerce",
                typical_audience="European dropshippers",
                relevance_keywords=["dropshipping", "europe", "supplier", "shipping"],
            ),
            Subreddit(
                name="r/affiliate_marketing",
                category="E-commerce",
                typical_audience="Affiliate marketing practitioners",
                relevance_keywords=["affiliate", "affiliate marketing", "commission"],
            ),
            # Additional Finance/Payments
            Subreddit(
                name="r/credit",
                category="Finance",
                typical_audience="Credit building and finance",
                relevance_keywords=["credit", "credit score", "lending", "creditworthiness"],
            ),
            Subreddit(
                name="r/lending",
                category="Finance",
                typical_audience="Lending and loan professionals",
                relevance_keywords=["lending", "loans", "borrowing", "credit"],
            ),
            # Additional Consulting
            Subreddit(
                name="r/managementconsulting",
                category="General Business",
                typical_audience="Management consulting professionals",
                relevance_keywords=["management consulting", "consulting", "strategy"],
            ),
            # Additional Niche Communities
            Subreddit(
                name="r/CallCenterWorkers",
                category="HR/Recruiting",
                typical_audience="Call center operators and managers",
                relevance_keywords=["call center", "customer service", "support"],
            ),
            Subreddit(
                name="r/Accounting_jobs",
                category="Finance",
                typical_audience="Accounting professionals and job seekers",
                relevance_keywords=["accounting jobs", "accounting", "cpa", "finance"],
            ),
            Subreddit(
                name="r/ExcelFinance",
                category="Finance",
                typical_audience="Excel and finance professionals",
                relevance_keywords=["excel", "spreadsheets", "finance", "modeling"],
            ),
            Subreddit(
                name="r/WholesaleRealEstate",
                category="Real Estate",
                typical_audience="Wholesale real estate investors",
                relevance_keywords=["wholesale real estate", "wholesaling", "property"],
            ),
            Subreddit(
                name="r/SeasonalWork",
                category="HR/Recruiting",
                typical_audience="Seasonal job seekers",
                relevance_keywords=["seasonal work", "temporary jobs", "employment"],
            ),
            Subreddit(
                name="r/learnmachinelearning",
                category="AI/ML",
                typical_audience="Machine learning students and learners",
                relevance_keywords=["learn", "machine learning", "tutorial", "course"],
            ),
            Subreddit(
                name="r/deeplearning",
                category="AI/ML",
                typical_audience="Deep learning researchers and practitioners",
                relevance_keywords=["deep learning", "neural networks", "tensorflow"],
            ),
            Subreddit(
                name="r/tensorflow",
                category="AI/ML",
                typical_audience="TensorFlow users and developers",
                relevance_keywords=["tensorflow", "machine learning", "keras", "ml"],
            ),
            Subreddit(
                name="r/pytorch",
                category="AI/ML",
                typical_audience="PyTorch users and deep learning developers",
                relevance_keywords=["pytorch", "deep learning", "neural networks"],
            ),
            Subreddit(
                name="r/nlp",
                category="AI/ML",
                typical_audience="Natural language processing specialists",
                relevance_keywords=["nlp", "natural language", "text processing"],
            ),
            Subreddit(
                name="r/computervision",
                category="AI/ML",
                typical_audience="Computer vision engineers",
                relevance_keywords=["computer vision", "image processing", "cv"],
            ),
            # Additional niche communities
            Subreddit(
                name="r/slack",
                category="Productivity",
                typical_audience="Slack users and developers",
                relevance_keywords=["slack", "messaging", "communication", "app"],
            ),
            Subreddit(
                name="r/zapier",
                category="Productivity",
                typical_audience="Zapier automation users",
                relevance_keywords=["zapier", "automation", "workflow", "integration"],
            ),
            Subreddit(
                name="r/Airtable",
                category="Productivity",
                typical_audience="Airtable users and builders",
                relevance_keywords=["airtable", "database", "spreadsheet", "productivity"],
            ),
            Subreddit(
                name="r/loopback_io",
                category="Developer Tools",
                typical_audience="LoopBack API framework users",
                relevance_keywords=["loopback", "api", "nodejs", "framework"],
            ),
            Subreddit(
                name="r/graphql",
                category="Developer Tools",
                typical_audience="GraphQL developers",
                relevance_keywords=["graphql", "api", "query language", "data"],
            ),
            Subreddit(
                name="r/rest",
                category="Developer Tools",
                typical_audience="REST API developers",
                relevance_keywords=["rest", "restful", "api", "http"],
            ),
            Subreddit(
                name="r/laravel",
                category="Developer Tools",
                typical_audience="Laravel PHP framework users",
                relevance_keywords=["laravel", "php", "framework", "web"],
            ),
            Subreddit(
                name="r/django",
                category="Developer Tools",
                typical_audience="Django Python framework users",
                relevance_keywords=["django", "python", "web framework", "backend"],
            ),
            Subreddit(
                name="r/flask",
                category="Developer Tools",
                typical_audience="Flask Python framework users",
                relevance_keywords=["flask", "python", "web framework", "microframework"],
            ),
            Subreddit(
                name="r/rails",
                category="Developer Tools",
                typical_audience="Ruby on Rails developers",
                relevance_keywords=["rails", "ruby", "web framework", "backend"],
            ),
            Subreddit(
                name="r/elasticsearch",
                category="Data",
                typical_audience="Elasticsearch and search specialists",
                relevance_keywords=["elasticsearch", "search", "logging", "kibana"],
            ),
            Subreddit(
                name="r/mongodb",
                category="Data",
                typical_audience="MongoDB users and developers",
                relevance_keywords=["mongodb", "nosql", "database", "json"],
            ),
            Subreddit(
                name="r/postgresql",
                category="Data",
                typical_audience="PostgreSQL users",
                relevance_keywords=["postgresql", "postgres", "sql", "database"],
            ),
            Subreddit(
                name="r/redis",
                category="Data",
                typical_audience="Redis cache users",
                relevance_keywords=["redis", "cache", "database", "performance"],
            ),
            Subreddit(
                name="r/rabbitmq",
                category="Developer Tools",
                typical_audience="RabbitMQ and messaging queue users",
                relevance_keywords=["rabbitmq", "message queue", "amqp", "messaging"],
            ),
            Subreddit(
                name="r/celery",
                category="Developer Tools",
                typical_audience="Celery task queue users",
                relevance_keywords=["celery", "task queue", "python", "async"],
            ),
            Subreddit(
                name="r/microservices",
                category="Developer Tools",
                typical_audience="Microservices architecture specialists",
                relevance_keywords=["microservices", "architecture", "services", "scalability"],
            ),
            Subreddit(
                name="r/serverless",
                category="Developer Tools",
                typical_audience="Serverless and FaaS developers",
                relevance_keywords=["serverless", "lambda", "faas", "functions"],
            ),
            Subreddit(
                name="r/terraform",
                category="Developer Tools",
                typical_audience="Terraform infrastructure code users",
                relevance_keywords=["terraform", "infrastructure", "iac", "deployment"],
            ),
            Subreddit(
                name="r/ansible",
                category="Developer Tools",
                typical_audience="Ansible automation users",
                relevance_keywords=["ansible", "automation", "deployment", "configuration"],
            ),
            Subreddit(
                name="r/monitoring",
                category="Developer Tools",
                typical_audience="Monitoring and observability specialists",
                relevance_keywords=["monitoring", "observability", "alerts", "metrics"],
            ),
            Subreddit(
                name="r/logging",
                category="Developer Tools",
                typical_audience="Logging and log management specialists",
                relevance_keywords=["logging", "logs", "debugging", "tracing"],
            ),
            Subreddit(
                name="r/security",
                category="Developer Tools",
                typical_audience="Cybersecurity and application security",
                relevance_keywords=["security", "cybersecurity", "encryption", "vulnerability"],
            ),
            Subreddit(
                name="r/appsecurity",
                category="Developer Tools",
                typical_audience="Application security specialists",
                relevance_keywords=["application security", "appsec", "vulnerability", "security"],
            ),
            Subreddit(
                name="r/testing",
                category="Developer Tools",
                typical_audience="Software testing and QA professionals",
                relevance_keywords=["testing", "qa", "quality assurance", "test automation"],
            ),
            Subreddit(
                name="r/leaderboards",
                category="Gaming",
                typical_audience="Competitive gaming and esports",
                relevance_keywords=["leaderboards", "esports", "competitive", "gaming"],
            ),
            Subreddit(
                name="r/AsianFinance",
                category="Finance",
                typical_audience="Asian financial markets and investments",
                relevance_keywords=["asian finance", "investment", "markets", "trading"],
            ),
            Subreddit(
                name="r/RealEstateMarketing",
                category="Real Estate",
                typical_audience="Real estate marketing specialists",
                relevance_keywords=["real estate marketing", "marketing", "real estate"],
            ),
        ]

    def recommend(self, icp_text: str) -> RecommendationResult:
        """
        Generate recommendations for an ICP.

        Args:
            icp_text: Plain text description of the ICP
                     (e.g., "CRM for real estate agents")

        Returns:
            RecommendationResult with subreddits, keywords, platforms, and summary
        """
        if not icp_text or not icp_text.strip():
            return RecommendationResult(
                subreddits=[],
                keywords=[],
                platforms=[],
                icp_summary="",
            )

        # Step 1: Extract and normalize ICP text
        icp_clean = icp_text.strip().lower()
        icp_summary = icp_text.strip()

        # Step 2: Generate keywords from ICP (with stopword filtering and concept expansion)
        keywords = self._generate_keywords(icp_clean)

        # Step 3: Score and rank subreddits (only using core keywords, not pain phrases)
        scored_subs = self._score_subreddits(keywords)

        # Step 4: Recommend platforms based on detected categories
        platforms = self._recommend_platforms(icp_clean, keywords, scored_subs)

        # Format results
        subreddit_list = [
            {
                "name": sub["name"],
                "relevance": round(sub["relevance"], 2),
                "reason": sub["reason"],
            }
            for sub in scored_subs
        ]

        return RecommendationResult(
            subreddits=subreddit_list,
            keywords=keywords,
            platforms=platforms,
            icp_summary=icp_summary,
        )

    def _generate_keywords(self, icp_text: str) -> list[str]:
        """
        Generate keywords from ICP text with stopword filtering and concept expansion.

        Includes:
        - Original ICP terms (filtered through stopwords)
        - Expanded concepts (abbreviations → full meanings)
        - Competitor names (if applicable)
        """
        keywords = []

        # Step 1: Extract base terms from original ICP text, filter stopwords
        original_terms = icp_text.split()
        base_terms = []
        for term in original_terms:
            term_clean = re.sub(r"[^\w\-]", "", term).lower()
            if term_clean and term_clean not in STOPWORDS and len(term_clean) > 2:
                base_terms.append(term_clean)

        # Add unique base terms (deduplicated)
        seen = set()
        for term in base_terms:
            if term not in seen and len(keywords) < 12:
                keywords.append(term)
                seen.add(term)

        # Step 2: Expand concepts — add compound phrases as whole keywords
        # These go into the keyword list as-is (not split into words)
        icp_lower = icp_text.lower()
        for concept, expansions in CONCEPT_SYNONYMS.items():
            if concept in icp_lower:
                for phrase in expansions:
                    phrase_clean = phrase.lower().strip()
                    if phrase_clean not in seen:
                        keywords.append(phrase_clean)
                        seen.add(phrase_clean)

        # Step 3: Add competitor suggestions (if applicable)
        for niche, competitors in self.COMPETITORS.items():
            if any(term in icp_text for term in niche.split()):
                keywords.extend(competitors[:3])
                keywords.extend([f"alternative to {c.lower()}" for c in competitors[:2]])
                break

        # Step 4: Deduplicate and return
        return list(dict.fromkeys(keywords))[:15]

    def _score_subreddits(self, keywords: list[str]) -> list[dict]:
        """
        Score and rank subreddits based on keyword overlap.

        Rules:
        - Requires at least 1 keyword match for consideration
        - Uses TF-IDF-style scoring: matches / subreddit_keyword_count
        - Applies minimum relevance threshold (0.15)
        - Avoids generic single-word matches by requiring longer matches or multi-word overlap
        - For multi-word relevance keywords, require full phrase match or word-boundary match
        """
        # Generic words that get half-weight (0.5 match) instead of full (1.0)
        generic_words = {"tool", "service", "system", "platform", "software", "app", "data"}

        scores = []

        for sub in self.subreddits:
            matches = 0
            matched_keywords = []

            for keyword in keywords:
                keyword_lower = keyword.lower()

                is_generic = keyword_lower in generic_words

                for rel_kw in sub.relevance_keywords:
                    rel_kw_lower = rel_kw.lower()

                    match_weight = 0.5 if is_generic else 1.0

                    # Exact match (handles both single and compound phrases)
                    if keyword_lower == rel_kw_lower:
                        matches += match_weight
                        matched_keywords.append(keyword)
                        break

                    # Multi-word keyword matches multi-word rel_kw (bidirectional containment)
                    if " " in keyword_lower and " " in rel_kw_lower:
                        if keyword_lower in rel_kw_lower or rel_kw_lower in keyword_lower:
                            matches += match_weight
                            matched_keywords.append(keyword)
                            break

                    # Single keyword appears as word within multi-word relevance keyword
                    if " " in rel_kw_lower:
                        rel_kw_words = rel_kw_lower.split()
                        if keyword_lower in rel_kw_words:
                            matches += match_weight
                            matched_keywords.append(keyword)
                            break

                    # Multi-word keyword contains single-word relevance keyword
                    if " " in keyword_lower and " " not in rel_kw_lower:
                        kw_words = keyword_lower.split()
                        if rel_kw_lower in kw_words and len(rel_kw_lower) >= 5:
                            matches += match_weight
                            matched_keywords.append(keyword)
                            break

                    # Single-word relevance keywords: require min length to avoid spurious matches
                    if " " not in keyword_lower and " " not in rel_kw_lower:
                        if len(keyword_lower) >= 5 and keyword_lower in rel_kw_lower:
                            matches += match_weight
                            matched_keywords.append(keyword)
                            break

            # Only consider subreddits with 1+ matches
            if matches >= 1:
                relevance = min(matches / max(len(sub.relevance_keywords), 1), 1.0)

                # Apply minimum threshold
                if relevance >= 0.15:
                    reason = f"Matches {matches} keywords: {', '.join(set(matched_keywords[:3]))}"

                    scores.append(
                        {
                            "name": sub.name,
                            "category": sub.category,
                            "relevance": relevance,
                            "reason": reason,
                            "audience": sub.typical_audience,
                        }
                    )

        # Sort by relevance (descending)
        scores.sort(key=lambda x: x["relevance"], reverse=True)

        # Return top 15
        return scores[:15]

    def _recommend_platforms(self, icp_text: str, keywords: list[str], scored_subs: list[dict]) -> list[str]:
        """
        Recommend platforms beyond Reddit based on ICP characteristics and detected categories.

        Platforms: github, stackoverflow, hn, twitter, producthunt, indiehackers
        """
        platforms = ["reddit"]  # Always include Reddit

        # Detect category from scored subreddits
        categories = set(sub.get("category", "") for sub in scored_subs)

        # Developer-focused
        dev_keywords = ["developer", "api", "programming", "engineer", "code", "devops", "webscraping"]
        is_developer_focused = any(kw in icp_text for kw in dev_keywords) or "Developer Tools" in categories
        if is_developer_focused:
            platforms.extend(["github", "stackoverflow", "hn"])

        # Founder/startup focused
        founder_keywords = ["founder", "startup", "startup", "indie", "bootstrapped", "0-to-1"]
        is_founder_focused = any(kw in icp_text for kw in founder_keywords) or any(
            cat in categories for cat in ["SaaS", "Entrepreneurship"]
        )
        if is_founder_focused:
            platforms.extend(["hn", "indiehackers", "producthunt"])

        # Marketing/growth focused
        marketing_keywords = ["marketing", "growth", "analytics", "content", "social media"]
        is_marketing_focused = any(kw in icp_text for kw in marketing_keywords) or "Marketing" in categories
        if is_marketing_focused:
            platforms.extend(["twitter", "producthunt"])

        # General tech/AI/SaaS → HN
        if any(kw in icp_text for kw in ["software", "tech", "ai", "saas", "artificial intelligence"]):
            if "hn" not in platforms:
                platforms.append("hn")

        # If broad or vague ICP, recommend all major platforms
        if not scored_subs or (len(keywords) < 3 and len(scored_subs) < 5):
            platforms = ["reddit", "github", "stackoverflow", "hn", "twitter", "producthunt", "indiehackers"]

        # Remove duplicates while preserving order
        return list(dict.fromkeys(platforms))


def recommend_for_icp(icp_text: str) -> RecommendationResult:
    """Convenience function: instantiate and recommend in one call."""
    recommender = Recommender()
    return recommender.recommend(icp_text)
