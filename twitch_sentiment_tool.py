#!/usr/bin/env python3
"""
Twitch Real-Time Chat Sentiment Analyzer v2.1
- Per-word analysis + semantic embedding clustering
- Sentence-level semantic clustering
- Configurable embedding models and similarity thresholds
- Toggleable @mention and !command filtering
"""

import socket
import re
import time
import argparse
import threading
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# ===================== CONFIG =====================
IRC_HOST = "irc.chat.twitch.tv"
IRC_PORT = 6667
PING_INTERVAL = 60
TOP_N = 10
SLIDING_WINDOW_SECONDS = 300
PRINT_INTERVAL_SECONDS = 8
DECAY_HALF_LIFE_SECONDS = 45

# Defaults that can be overridden by CLI args
DEFAULT_WORD_SIMILARITY_THRESHOLD = 0.72
DEFAULT_SENTENCE_SIMILARITY_THRESHOLD = 0.78
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"

# Global holders for models
embed_model = None
sentiment_pipeline = None

def load_models(model_name):
    global embed_model, sentiment_pipeline
    print(f"[INFO] Loading embedding model: {model_name}...")
    embed_model = SentenceTransformer(model_name)
    print("[INFO] Loading sentiment model...")
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    print("[INFO] Models ready.\n")

STOPWORDS = {"the","a","an","and","or","but","in","on","at","to","for","of","with","is","are","was","were","be","been","being","have","has","had","do","does","did","will","would","could","should","can","this","that","these","those","i","you","he","she","it","we","they","me","him","her","us","them","my","your","his","her","its","our","their","so","just","like","really","very","much","now","here","there","when","where","why","how","all","any","some","no","not","yes","yeah","lol","lmao","omg","wtf","gg","ez","pog","poggers","kappa","monka","pepe","feels","good","bad","nice","love","hate"}

def tokenize_words(text, min_len=3, ignore_words=None, ignore_mentions=True):
    pattern = rf'\b[a-zA-Z]{{{min_len},}}\b'
    words = re.findall(pattern, text.lower())
    ignore_set = ignore_words or set()
    result = []
    for w in words:
        if w in STOPWORDS:
            continue
        if ignore_mentions and w.startswith('@'):
            continue
        if w in ignore_set:
            continue
        result.append(w)
    return result

def get_word_sentiment(word, sentence_score):
    try:
        res = sentiment_pipeline(word[:128])[0]
        score = float(res["score"])
        if res["label"].upper() == "NEGATIVE":
            score = -score
        return 0.7 * score + 0.3 * sentence_score
    except:
        return 0.0

class SemanticSentimentTracker:
    def __init__(self, min_word_len=3, min_sentence_words=2, ignore_words=None, 
                 word_threshold=DEFAULT_WORD_SIMILARITY_THRESHOLD, 
                 sent_threshold=DEFAULT_SENTENCE_SIMILARITY_THRESHOLD):
        self.clusters = []
        self.sentence_clusters = []
        self.lock = threading.Lock()
        self.message_count = 0
        self.min_word_len = min_word_len
        self.min_sentence_words = min_sentence_words
        self.ignore_words = ignore_words or set()
        self.word_threshold = word_threshold
        self.sent_threshold = sent_threshold

    def _get_embedding(self, text):
        return embed_model.encode(text, convert_to_tensor=True, show_progress_bar=False)

    def add_message(self, text, ts=None, ignore_mentions=True, ignore_commands=True):
        if ts is None:
            ts = time.time()

        if ignore_commands and text.strip().startswith('!'):
            return

        try:
            sent_res = sentiment_pipeline(text[:512])[0]
            sent_score = float(sent_res["score"])
            if sent_res["label"].upper() == "NEGATIVE":
                sent_score = -sent_score
        except:
            sent_score = 0.0

        words = tokenize_words(text, self.min_word_len, self.ignore_words, ignore_mentions)
        if len(words) < self.min_sentence_words:
            return
        words = list(dict.fromkeys(words))

        with self.lock:
            self.message_count += 1

            # Word-level clustering
            for word in words:
                word_score = get_word_sentiment(word, sent_score)
                emb = self._get_embedding(word)
                matched = False
                for cluster in self.clusters:
                    if float(util.cos_sim(emb, cluster["embedding"])) >= self.word_threshold:
                        age = ts - cluster["last_ts"]
                        freshness = 2 ** (-age / DECAY_HALF_LIFE_SECONDS)
                        weighted = abs(word_score) * freshness
                        cluster["total_score"] += weighted
                        cluster["count"] += 1
                        cluster["last_ts"] = ts
                        if word not in cluster["members"]:
                            cluster["members"].append(word)
                        if ts >= cluster.get("last_ts", 0):
                            cluster["best_sentence"] = text
                        matched = True
                        break
                if not matched:
                    self.clusters.append({
                        "rep_word": word, "embedding": emb, "total_score": abs(word_score),
                        "count": 1, "last_ts": ts, "members": [word], "best_sentence": text
                    })

            # Sentence-level clustering
            sent_emb = self._get_embedding(text)
            matched_sent = False
            for scluster in self.sentence_clusters:
                if float(util.cos_sim(sent_emb, scluster["embedding"])) >= self.sent_threshold:
                    age = ts - scluster["last_ts"]
                    freshness = 2 ** (-age / DECAY_HALF_LIFE_SECONDS)
                    scluster["total_score"] += abs(sent_score) * freshness
                    scluster["count"] += 1
                    scluster["last_ts"] = ts
                    if len(scluster["examples"]) < 3:
                        scluster["examples"].append(text)
                    matched_sent = True
                    break
            if not matched_sent:
                self.sentence_clusters.append({
                    "embedding": sent_emb,
                    "total_score": abs(sent_score),
                    "count": 1,
                    "last_ts": ts,
                    "examples": [text]
                })

            cutoff = ts - SLIDING_WINDOW_SECONDS
            self.clusters = [c for c in self.clusters if c["last_ts"] >= cutoff]
            self.sentence_clusters = [c for c in self.sentence_clusters if c["last_ts"] >= cutoff]

    def get_top_word_sentiments(self, n=TOP_N):
        now = time.time()
        ranked = []
        with self.lock:
            for c in self.clusters:
                age = now - c["last_ts"]
                freshness = 2 ** (-age / DECAY_HALF_LIFE_SECONDS)
                final_score = c["total_score"] * freshness
                ranked.append({
                    "type": "word",
                    "sentiment": c["rep_word"],
                    "score": round(final_score, 3),
                    "count": c["count"],
                    "members": ", ".join(c["members"][:5]),
                    "sentence": c.get("best_sentence", c["rep_word"]),
                    "freshness": "🆕" if age < 30 else "  ",
                    "last_seen": datetime.fromtimestamp(c["last_ts"]).strftime("%H:%M:%S")
                })
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:n]

    def get_top_sentence_sentiments(self, n=TOP_N):
        now = time.time()
        ranked = []
        with self.lock:
            for c in self.sentence_clusters:
                age = now - c["last_ts"]
                freshness = 2 ** (-age / DECAY_HALF_LIFE_SECONDS)
                final_score = c["total_score"] * freshness
                ranked.append({
                    "type": "sentence",
                    "sentiment": c["examples"][0][:60] + ("..." if len(c["examples"][0]) > 60 else ""),
                    "score": round(final_score, 3),
                    "count": c["count"],
                    "members": "",
                    "sentence": c["examples"][0],
                    "freshness": "🆕" if age < 30 else "  ",
                    "last_seen": datetime.fromtimestamp(c["last_ts"]).strftime("%H:%M:%S")
                })
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:n]

    def status(self):
        with self.lock:
            return self.message_count, len(self.clusters), len(self.sentence_clusters)

def connect_and_listen(channel, token, ignore_users, min_word_len, min_sentence_words, ignore_words,
                       ignore_mentions, ignore_commands, word_threshold, sent_threshold):
    ignore_set = {u.lower().strip() for u in ignore_users} if ignore_users else set()
    ignore_word_set = {w.lower().strip() for w in ignore_words} if ignore_words else set()

    nick = "justinfan" + str(int(time.time()) % 100000)
    if not token.startswith("oauth:"):
        token = "oauth:\" + token

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(300)
    print(f"[INFO] Connecting to Twitch IRC for #{channel} ...")
    sock.connect((IRC_HOST, IRC_PORT))
    sock.send(f"PASS {token}\r\n".encode("utf-8"))
    sock.send(f"NICK {nick}\r\n".encode("utf-8"))
    sock.send(f"JOIN #{channel}\r\n".encode("utf-8"))
    print(f"[INFO] Joined #{channel}. Per-word + sentence-level semantic clustering active.")
    if ignore_set: print(f"[INFO] Ignoring users: {', '.join(ignore_set)}")
    if ignore_word_set: print(f"[INFO] Ignoring words: {', '.join(ignore_word_set)}")
    print(f"[INFO] Ignore @mentions: {ignore_mentions} | Ignore !commands: {ignore_commands}")
    print(f"[INFO] Min word len: {min_word_len} | Min words/sentence: {min_sentence_words}")
    print(f"[INFO] Word Similarity Threshold: {word_threshold} | Sentence Similarity Threshold: {sent_threshold}\n")

    tracker = SemanticSentimentTracker(min_word_len, min_sentence_words, ignore_word_set, word_threshold, sent_threshold)
    last_ping = time.time()

    def printer():
        while True:
            time.sleep(PRINT_INTERVAL_SECONDS)
            word_top = tracker.get_top_word_sentiments()
            sent_top = tracker.get_top_sentence_sentiments()
            msg_count, w_clusters, s_clusters = tracker.status()

            print(f"\n=== TOP {TOP_N} WORD CLUSTERS ===")
            if not word_top:
                print("  (No word clusters yet)")
            else:
                for i, s in enumerate(word_top, 1):
                    print(f"{i:2}. {s['freshness']} {s['sentiment']:<16} score={s['score']:.3f} count={s['count']}")
                    print(f"    Sentence: \"{s['sentence']}\"  | members=[{s['members']}]")

            print(f"\n=== TOP {TOP_N} SIMILAR SENTENCE CLUSTERS ===")
            if not sent_top:
                print("  (No similar sentence groups yet)")
            else:
                for i, s in enumerate(sent_top, 1):
                    print(f"{i:2}. {s['freshness']} score={s['score']:.3f} count={s['count']}  last={s['last_seen']}")
                    print(f"    \"{s['sentence']}\"")

            print(f"\nTotal messages: {msg_count} | Word clusters: {w_clusters} | Sentence clusters: {s_clusters}")
            print("=" * 95)

    threading.Thread(target=printer, daemon=True).start()

    buffer = ""
    try:
        while True:
            data = sock.recv(2048).decode("utf-8", errors="ignore")
            if not data: break
            buffer += data
            lines = buffer.split("\r\n")
            buffer = lines[-1]
            for line in lines[:-1]:
                line = line.strip()
                if not line: continue
                if line.startswith("PING"):
                    sock.send("PONG :tmi.twitch.tv\r\n".encode("utf-8"))
                    last_ping = time.time()
                    continue
                match = re.match(r":(\S+)!\S+@\S+\.tmi\.twitch\.tv PRIVMSG #\S+ :(.*)", line)
                if match:
                    user = match.group(1).lower()
                    if user in ignore_set: continue
                    msg = match.group(2).strip()
                    if msg:
                        tracker.add_message(msg, ignore_mentions=ignore_mentions, ignore_commands=ignore_commands)
            if time.time() - last_ping > PING_INTERVAL:
                sock.send("PING :tmi.twitch.tv\r\n".encode("utf-8"))
                last_ping = time.time()
    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
    finally:
        sock.close()
        print("[INFO] Disconnected.")

def main():
    parser = argparse.ArgumentParser(description="Twitch chat sentiment analyzer")
    parser.add_argument("--channel", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--ignore-users", default="")
    parser.add_argument("--ignore-words", default="")
    parser.add_argument("--min-word-len", type=int, default=3)
    parser.add_argument("--min-sentence-words", type=int, default=2)
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL, help=f"Embedding model to use (default: {DEFAULT_EMBED_MODEL})")
    parser.add_argument("--word-threshold", type=float, default=DEFAULT_WORD_SIMILARITY_THRESHOLD, help=f"Similarity threshold for word clustering (default: {DEFAULT_WORD_SIMILARITY_THRESHOLD})")
    parser.add_argument("--sent-threshold", type=float, default=DEFAULT_SENTENCE_SIMILARITY_THRESHOLD, help=f"Similarity threshold for sentence clustering (default: {DEFAULT_SENTENCE_SIMILARITY_THRESHOLD})")
    parser.add_argument("--ignore-mentions", action="store_true", default=True,
                        help="Ignore words starting with @ (default: True)")
    parser.add_argument("--no-ignore-mentions", dest="ignore_mentions", action="store_false",
                        help="Do not ignore @mentions")
    parser.add_argument("--ignore-commands", action="store_true", default=True,
                        help="Ignore messages starting with ! (default: True)")
    parser.add_argument("--no-ignore-commands", dest="ignore_commands", action="store_false",
                        help="Do not ignore !commands")
    args = parser.parse_args()

    load_models(args.model)

    ignore_users = [u.strip() for u in args.ignore_users.split(",") if u.strip()]
    ignore_words = [w.strip() for w in args.ignore_words.split(",") if w.strip()]

    connect_and_listen(
        args.channel.lower(), args.token, ignore_users,
        args.min_word_len, args.min_sentence_words, ignore_words,
        args.ignore_mentions, args.ignore_commands, args.word_threshold, args.sent_threshold
    )

if __name__ == "__main__":
    main()