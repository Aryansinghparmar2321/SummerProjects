
"""
chatbot.py
-----------
Project 6: Chatbot for Customer Service
 
Implements TWO approaches as suggested in the brief:
 
1. Rule-based mode: simple keyword/substring matching against known
   customer-query patterns. Fast, transparent, zero dependencies beyond
   the standard library.
 
2. Simple NLP-based mode: TF-IDF vectorization + cosine similarity to
   match the user's message against known example patterns, so it can
   handle rephrasing that isn't an exact keyword hit (e.g. "my card got
   declined" still matches the payment_issue intent even though it
   shares almost no exact words with the training patterns).
 
Both modes share the same intents.json knowledge base, so you can compare
them directly -- a good way to actually SEE the difference between
rule-based and NLP-based chatbot design, which is the point of this
project.
 
Usage:
    python chatbot.py            # interactive chat, NLP mode (default)
    python chatbot.py --mode rule   # interactive chat, rule-based mode
    python chatbot.py --demo     # run both modes over sample queries, no input needed
"""
 
import json
import random
import re
import argparse
 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
 
def load_intents(path="intents.json"):
    with open(path, "r") as f:
        return json.load(f)
 
 
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "am",
    "i", "you", "your", "my", "me", "of", "to", "for", "in", "on", "at",
    "and", "or", "but", "do", "does", "did", "have", "has", "had",
    "this", "that", "it", "its", "with", "can", "could", "would", "will",
    "what", "when", "where", "who", "how", "please", "so", "get", "got"
}
 
 
def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text
 
 
def content_words(text):
    """Words used for overlap scoring, with stopwords removed so generic
    words like 'what'/'is'/'my' don't trigger false-positive intent matches."""
    return {w for w in text.split() if w not in STOPWORDS}
 
 
# ---------------------------------------------------------------------------
# Approach 1: Rule-based chatbot
# ---------------------------------------------------------------------------
class RuleBasedChatbot:
    """
    Matches user input against intent patterns using simple keyword/substring
    overlap. Picks the intent whose patterns share the most words with the
    user's message. Falls back to a generic response if nothing scores above
    a minimum threshold.
    """
 
    def __init__(self, intents_data, min_overlap=2):
        self.intents = intents_data["intents"]
        self.fallback_responses = intents_data["fallback_responses"]
        self.min_overlap = min_overlap
 
    def get_response(self, message):
        norm_msg = normalize(message)
        msg_words = content_words(norm_msg)
 
        best_tag = None
        best_score = 0
 
        for intent in self.intents:
            for pattern in intent["patterns"]:
                pattern_words = content_words(normalize(pattern))
                overlap = len(msg_words & pattern_words)
                # exact phrase match (whole-word boundaries) is a very strong signal
                if re.search(r"\b" + re.escape(normalize(pattern)) + r"\b", norm_msg):
                    overlap += 3
                if overlap > best_score:
                    best_score = overlap
                    best_tag = intent["tag"]
 
        if best_tag and best_score >= self.min_overlap:
            intent = next(i for i in self.intents if i["tag"] == best_tag)
            return random.choice(intent["responses"]), best_tag, best_score
        else:
            return random.choice(self.fallback_responses), "fallback", best_score
 
 
# ---------------------------------------------------------------------------
# Approach 2: Simple NLP-based chatbot (TF-IDF + cosine similarity)
# ---------------------------------------------------------------------------
class NLPChatbot:
    """
    Builds a TF-IDF vector space over ALL example patterns across all
    intents. A new message is vectorized the same way, and we find the
    training pattern with the highest cosine similarity to it. This
    generalizes better than plain keyword overlap because TF-IDF downweights
    common filler words and rewards distinctive, topic-specific ones.
    """
 
    def __init__(self, intents_data, similarity_threshold=0.3):
        self.intents = intents_data["intents"]
        self.fallback_responses = intents_data["fallback_responses"]
        self.similarity_threshold = similarity_threshold
 
        self.pattern_texts = []
        self.pattern_tags = []
        for intent in self.intents:
            for pattern in intent["patterns"]:
                self.pattern_texts.append(normalize(pattern))
                self.pattern_tags.append(intent["tag"])
 
        self.vectorizer = TfidfVectorizer(stop_words=list(STOPWORDS))
        self.pattern_vectors = self.vectorizer.fit_transform(self.pattern_texts)
 
    def get_response(self, message):
        norm_msg = normalize(message)
        msg_vector = self.vectorizer.transform([norm_msg])
        similarities = cosine_similarity(msg_vector, self.pattern_vectors)[0]
 
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]
 
        if best_score >= self.similarity_threshold:
            best_tag = self.pattern_tags[best_idx]
            intent = next(i for i in self.intents if i["tag"] == best_tag)
            return random.choice(intent["responses"]), best_tag, round(float(best_score), 3)
        else:
            return random.choice(self.fallback_responses), "fallback", round(float(best_score), 3)
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run_interactive(bot, mode_name):
    print(f"\n=== Customer Service Chatbot ({mode_name} mode) ===")
    print("Type 'quit' to exit.\n")
    print("Bot: Hello! Welcome to support. How can I help you today?")
    while True:
        try:
            user_input = input("You: ")
        except EOFError:
            break
        if user_input.strip().lower() in ("quit", "exit"):
            print("Bot: Goodbye!")
            break
        response, tag, score = bot.get_response(user_input)
        print(f"Bot: {response}")
 
 
def run_demo(rule_bot, nlp_bot):
    sample_queries = [
        "Hi there",
        "where's my package",
        "my card got declined",              # rephrased -- no exact keyword overlap with patterns
        "can I send this back for a refund",  # rephrased return request
        "what time do you guys open",
        "I need to talk to a real person",
        "what's the meaning of life",         # should hit fallback
    ]
    print("\n=== DEMO: Rule-based vs NLP-based responses ===\n")
    for q in sample_queries:
        rule_resp, rule_tag, rule_score = rule_bot.get_response(q)
        nlp_resp, nlp_tag, nlp_score = nlp_bot.get_response(q)
        print(f"User: {q}")
        print(f"  [Rule-based]  intent={rule_tag:<20} score={rule_score:<5} -> {rule_resp}")
        print(f"  [NLP-based]   intent={nlp_tag:<20} score={nlp_score:<5} -> {nlp_resp}")
        print()
 
 
def main():
    parser = argparse.ArgumentParser(description="Customer Service Chatbot")
    parser.add_argument("--mode", choices=["rule", "nlp"], default="nlp",
                         help="Chatbot engine to use in interactive mode (default: nlp)")
    parser.add_argument("--demo", action="store_true",
                         help="Run both engines over sample queries and exit (no input needed)")
    args = parser.parse_args()
 
    intents_data = load_intents()
    rule_bot = RuleBasedChatbot(intents_data)
    nlp_bot = NLPChatbot(intents_data)
 
    if args.demo:
        run_demo(rule_bot, nlp_bot)
        return
 
    bot = rule_bot if args.mode == "rule" else nlp_bot
    run_interactive(bot, "rule-based" if args.mode == "rule" else "NLP-based (TF-IDF)")
 
 
if __name__ == "__main__":
    main()