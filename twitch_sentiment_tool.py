#!/usr/bin/env python3
"""
Twitch Real-Time Chat Sentiment Analyzer v2.4
- Per-word analysis + semantic embedding clustering
- Sentence-level semantic clustering
- Configurable embedding models and similarity thresholds
- Toggleable @mention and !command filtering
- IRC reader in a daemon thread (true decoupling via temp file or in-memory queue)
- Bounded queue (back‑pressure) between reader and consumer
- Configurable decay half-life
- Regex-based ignore filter
- Automatic fetching of third‑party emotes (BTTV, FFZ, 7TV) to exclude from analysis
- Graceful shutdown
"""

import sys
import socket
import re
import time
import argparse
import json
import os
import queue
import urllib.request
from datetime import datetime
from threading import Lock, Event, Thread
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# ===================== CONFIG =====================
IRC_HOST = "irc.chat.twitch.tv"
IRC_PORT = 6667
PING_INTERVAL = 60
TOP_N = 10
SLIDING_WINDOW_SECONDS = 300
DEFAULT_DECAY_HALF_LIFE_SECONDS = 45
QUEUE_MAXSIZE = 1000  # maximum number of messages buffered in queue

# Defaults that can be overridden by CLI args
DEFAULT_WORD_SIMILARITY_THRESHOLD = 0.72
DEFAULT_SENTENCE_SIMILARITY_THRESHOLD = 0.78
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"

# Global holders for models (set in main process)
embed_model = None
sentiment_pipeline = None

STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could","should","can","this","that","these","those","i","you","he","she","it",
    "we","they","me","him","her","us","them","my","your","his","her","its","our","their","so","just","like","really","very","much",
    "now","here","there","when","where","why","how","all","any","some","no","not","yes","yeah"
}

# Common Twitch emotes/commands to exclude from sentiment analysis
TWITCH_EMOTES = {
    "mods","kurwa","pog","poggers","kappa","monka","pepe","feels",
    "lul","omegalul","keks","widepeepo","dendi","ayaya","dentge",
    "xdd","prayge","monkas","sadge","pepelaugh","pepela",
    "weirdchamp","pogchamp","pogyou","page","weeg","snark",
    "susge","icant","goodone","catjam","rainbowpls","kekw",
    "clap","widepeepohappy","peepohappy","peeposad","weirdginger",
    "kapp"
}

# Helper functions to fetch third‑party emotes
def fetch_twitch_user_id(login):
    """Resolve a Twitch login name to a user ID using decapi (public, no auth)."""
    try:
        url = f"https://decapi.me/twitch/id/{login}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.read().decode('utf-8').strip()
    except Exception:
        return None

def fetch_bttv_emotes(user_id):
    """Fetch channel emotes from BetterTTV (v3) given a Twitch user ID."""
    emotes = set()
    if not user_id:
        return emotes
    try:
        url = f"https://api.betterttv.net/3/cached/users/twitch/{user_id}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            for eset in data.get('emoteSets', []):
                for e in eset.get('emotes', []):
                    code = e.get('code', '').lower()
                    if code:
                        emotes.add(code)
    except Exception:
        pass
    return emotes

def fetch_ffz_emotes(username):
    """Fetch channel emotes from FrankerFaceZ (v1) given a username."""
    emotes = set()
    try:
        url = f"https://api.frankerfacez.com/v1/room/{username}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            for eset in data.get('sets', {}).values():
                for e in eset.get('emoticons', []):
                    name = e.get('name', '').lower()
                    if name:
                        emotes.add(name)
    except Exception:
        pass
    return emotes

def fetch_7tv_emotes(username):
    """Fetch channel emotes from 7TV (v2) given a username."""
    emotes = set()
    try:
        url = f"https://api.7tv.app/v2/users/{username}/emotes"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            for e in data:
                name = e.get('name', '').lower()
                if name:
                    emotes.add(name)
    except Exception:
        pass
    return emotes

def fetch_channel_emotes(channel_name):
    """Fetch emotes from BTTV, FFZ, 7TV for the given channel name (lowercase)."""
    channel_name = channel_name.lower()
    print(f"[INFO] Fetching third‑party emotes for channel {channel_name} ...", flush=True)
    all_emotes = set()
    # Get Twitch user ID for BTTV
    user_id = fetch_twitch_user_id(channel_name)
    if user_id:
        print(f"[INFO] Twitch user ID: {user_id}", flush=True)
    else:
        print("[WARN] Could not resolve Twitch user ID; skipping BTTV emotes.", flush=True)
    # Fetch from each provider
    bttv = fetch_bttv_emotes(user_id)
    ffz = fetch_ffz_emotes(channel_name)
    _7tv = fetch_7tv_emotes(channel_name)
    all_emotes.update(bttv)
    all_emotes.update(ffz)
    all_emotes.update(_7tv)
    if all_emotes:
        print(f"[INFO] Fetched {len(all_emotes)} unique emotes from BTTV/FFZ/7TV.", flush=True)
    else:
        print("[INFO] No emotes fetched from third‑party providers.", flush=True)
    return all_emotes

def load_models(model_name: str):
    """Load embedding and sentiment models into global variables."""
    global embed_model, sentiment_pipeline
    print(f"[INFO] Loading embedding model: {model_name}...", flush=True)
    embed_model = SentenceTransformer(model_name)
    print("[INFO] Loading sentiment model...", flush=True)
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    print("[INFO] Models ready.\n", flush=True)

def tokenize_words(text, min_len=3, ignore_words=None, ignore_mentions=True):
    # Strip URLs before tokenization (P1)
    text = re.sub(r'https?://\S+', '', text)
    ignore_set = ignore_words or set()
    if ignore_mentions:
        # Filter @mentions before regex — split words, drop @-prefixed tokens (P0)
        tokens = text.lower().split()
        tokens = [t for t in tokens if not t.startswith('@')]
        text = ' '.join(tokens)
    pattern = rf'\b[a-zA-Z]{{{min_len},}}\b'
    words = re.findall(pattern, text.lower())
    result = []
    for w in words:
        if w in STOPWORDS:
            continue
        if w in TWITCH_EMOTES:
            continue
        if w in ignore_set:
            continue
        result.append(w)
    return result

def get_word_sentiment(word, sentence_score):
    # P4: No per-word pipeline call — reuse sentence-level score directly
    # Single words don't carry reliable sentiment from distilbert
    return sentence_score

class SemanticSentimentTracker:
    def __init__(self, min_word_len=3, min_sentence_words=2, ignore_words=None,
                 word_threshold=DEFAULT_WORD_SIMILARITY_THRESHOLD,
                 sent_threshold=DEFAULT_SENTENCE_SIMILARITY_THRESHOLD,
                 decay_half_life=DEFAULT_DECAY_HALF_LIFE_SECONDS):
        self.clusters = []
        self.sentence_clusters = []
        self.lock = Lock()
        self.message_count = 0
        self.ignored_regex_count = 0  # count of messages dropped by regex
        self.min_word_len = min_word_len
        self.min_sentence_words = min_sentence_words
        self.ignore_words = set(ignore_words) if ignore_words else set()
        self.word_threshold = float(word_threshold)
        self.sent_threshold = float(sent_threshold)
        self.decay_half_life = float(decay_half_life)
        # For variable refresh rate
        self.messages_at_last_print = 0

    def _get_embedding(self, text):
        # Defensive: if model not loaded yet, return zero tensor of expected size
        if embed_model is None:
            # Return a zero tensor of shape (384,) for all-MiniLM-L6-v2; safer to avoid crashes
            return torch.zeros(384)
        return embed_model.encode(text, convert_to_tensor=True, show_progress_bar=False)

    def add_message(self, text, ts=None, ignore_mentions=True, ignore_commands=True, ignore_regex=None):
        if ts is None:
            ts = time.time()

        if ignore_commands and text.strip().startswith('!'):
            return

        # Apply regex ignore if provided
        if ignore_regex is not None:
            try:
                if re.search(ignore_regex, text):
                    with self.lock:
                        self.ignored_regex_count += 1
                    return
            except re.error:
                pass  # invalid regex, ignore

        try:
            sent_res = sentiment_pipeline(text[:512])[0]
            sent_score = float(sent_res["score"])
            if sent_res["label"].upper() == "NEGATIVE":
                sent_score = -sent_score
        except Exception:
            sent_score = 0.0

        words = tokenize_words(text, self.min_word_len, self.ignore_words, ignore_mentions)

        with self.lock:
            self.message_count += 1

            # Always do sentence-level clustering for every chat message
            sent_emb = self._get_embedding(text)
            matched_sent = False
            for scluster in self.sentence_clusters:
                if float(util.cos_sim(sent_emb, scluster["embedding"])) >= self.sent_threshold:
                    age = ts - scluster["last_ts"]
                    decay = 2 ** (-age / self.decay_half_life)
                    # P2: Decay stored scores before accumulating
                    scluster["total_score"] = scluster["total_score"] * decay + abs(sent_score)
                    scluster["net_score"] = scluster["net_score"] * decay + sent_score
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
                    "net_score": sent_score,
                    "count": 1,
                    "last_ts": ts,
                    "examples": [text]
                })

            # Word-level clustering: only when enough meaningful words exist
            if len(words) >= self.min_sentence_words:
                words = list(dict.fromkeys(words))
                for word in words:
                    word_score = get_word_sentiment(word, sent_score)
                    emb = self._get_embedding(word)
                    matched = False
                    for cluster in self.clusters:
                        if float(util.cos_sim(emb, cluster["embedding"])) >= self.word_threshold:
                            age = ts - cluster["last_ts"]
                            decay = 2 ** (-age / self.decay_half_life)
                            # P2: Decay stored scores before accumulating
                            cluster["total_score"] = cluster["total_score"] * decay + abs(word_score)
                            cluster["net_score"] = cluster["net_score"] * decay + word_score
                            cluster["count"] += 1
                            cluster["last_ts"] = ts
                            if word not in cluster["members"]:
                                cluster["members"].append(word)
                            # Update best_sentence even from short-word messages (stored text is always full)
                            if ts >= cluster.get("last_ts", 0):
                                cluster["best_sentence"] = text
                            matched = True
                            break
                    if not matched:
                        self.clusters.append({
                            "rep_word": word,
                            "embedding": emb,
                            "total_score": abs(word_score),
                            "net_score": word_score,
                            "count": 1,
                            "last_ts": ts,
                            "members": [word],
                            "best_sentence": text
                        })

            # Sliding window cleanup
            cutoff = ts - SLIDING_WINDOW_SECONDS
            self.clusters = [c for c in self.clusters if c["last_ts"] >= cutoff]
            self.sentence_clusters = [c for c in self.sentence_clusters if c["last_ts"] >= cutoff]

    def _polarity_emoji(self, net_score):
        if net_score > 0.1:
            return "➕"
        elif net_score < -0.1:
            return "➖"
        return "⚪"

    def _polarity_label(self, net_score):
        if net_score > 0.1:
            return "pos"
        elif net_score < -0.1:
            return "neg"
        return "neu"

    def get_top_word_sentiments(self, n=TOP_N):
        now = time.time()
        ranked = []
        with self.lock:
            for c in self.clusters:
                age = now - c["last_ts"]
                freshness = 2 ** (-age / self.decay_half_life)
                final_score = c["total_score"] * freshness
                ranked.append({
                    "type": "word",
                    "sentiment": c["rep_word"],
                    "score": round(final_score, 3),
                    "count": c["count"],
                    "members": ", ".join(c["members"][:5]),
                    "sentence": c.get("best_sentence", c["rep_word"]),
                    "polarity": self._polarity_emoji(c["net_score"]),
                    "polarity_label": self._polarity_label(c["net_score"]),
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
                freshness = 2 ** (-age / self.decay_half_life)
                final_score = c["total_score"] * freshness
                ranked.append({
                    "type": "sentence",
                    "sentiment": c["examples"][0][:60] + ("..." if len(c["examples"][0]) > 60 else ""),
                    "score": round(final_score, 3),
                    "count": c["count"],
                    "members": "",
                    "sentence": c["examples"][0],
                    "polarity": self._polarity_emoji(c["net_score"]),
                    "polarity_label": self._polarity_label(c["net_score"]),
                    "freshness": "🆕" if age < 30 else "  ",
                    "last_seen": datetime.fromtimestamp(c["last_ts"]).strftime("%H:%M:%S")
                })
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:n]

    def status(self):
        with self.lock:
            return self.message_count, len(self.clusters), len(self.sentence_clusters), self.ignored_regex_count

# Added reconnection logic to IRC reader thread

def irc_reader_thread(channel, token, ignore_users_set, ignore_words_set,
                      ignore_mentions, ignore_commands, ignore_regex, 
                      q: queue.Queue, dropped_list):
    """
    Runs in a daemon thread: reads from Twitch IRC and puts raw messages into the queue.
    Never performs heavy NLP work.
    Uses a bounded queue for simple back‑pressure.
    Includes automatic reconnection with exponential back‑off on socket errors.
    """
    nick = "justinfan" + str(int(time.time()) % 100000)
    if not token.startswith("oauth:"):
        token = "oauth:" + token

    backoff = 1  # start with 1 second
    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(300)
        try:
            sock.connect((IRC_HOST, IRC_PORT))
            sock.send(f"PASS {token}\r\n".encode("utf-8"))
            sock.send(f"NICK {nick}\r\n".encode("utf-8"))
            sock.send(f"JOIN #{channel}\r\n".encode("utf-8"))
            buffer = ""
            last_ping = time.time()
            while True:
                try:
                    data = sock.recv(4096).decode("utf-8", errors="ignore")
                except socket.timeout:
                    if time.time() - last_ping > PING_INTERVAL:
                        sock.send(b"PING :tmi.twitch.tv\r\n")
                        last_ping = time.time()
                    continue
                if not data:
                    # Connection closed by server
                    raise ConnectionError("Socket closed")
                buffer += data
                lines = buffer.split("\r\n")
                buffer = lines[-1]
                for line in lines[:-1]:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("PING"):
                        sock.send(b"PONG :tmi.twitch.tv\r\n")
                        last_ping = time.time()
                        continue
                    match = re.match(r":(\S+)!\S+@\S+\.tmi\.twitch\.tv PRIVMSG #\S+ :(.*)", line)
                    if match:
                        user = match.group(1).lower()
                        if user in ignore_users_set:
                            continue
                        msg = match.group(2).strip()
                        if msg:
                            # Apply regex ignore if provided
                            if ignore_regex is not None:
                                try:
                                    if re.search(ignore_regex, msg):
                                        dropped_list[0] += 1
                                        continue
                                except re.error:
                                    pass
                            # Put with blocking wait for space (simple back‑pressure)
                            try:
                                q.put({"text": msg, "ts": time.time(), "user": user}, block=True, timeout=1.0)
                            except Exception:
                                dropped_list[0] += 1
                # Periodic ping if needed
                if time.time() - last_ping > PING_INTERVAL:
                    sock.send(b"PING :tmi.twitch.tv\r\n")
                    last_ping = time.time()
        except Exception as e:
            # Log and attempt reconnection after back‑off delay
            print(f"[WARN] IRC connection lost ({e}), retrying in {backoff}s...", flush=True)
            try:
                sock.close()
            except:
                pass
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)  # cap at 60 s
            continue
        else:
            # Normal exit (should not happen in daemon mode)
            break
    # Signal end of stream if exiting for any reason
    q.put(None)

def printer_thread(tracker, stop_event):
    """
    Runs in a daemon thread: reads from tracker and prints periodic updates.
    """
    while not stop_event.is_set():
        time.sleep(2)
        with tracker.lock:
            current_count = tracker.message_count
            new_messages = current_count - tracker.messages_at_last_print
        if new_messages >= 10:
            pass
        elif new_messages >= 5:
            pass
        elif new_messages >= 2:
            time.sleep(1)
        elif new_messages == 1:
            time.sleep(2)
        else:
            time.sleep(3)
        word_top = tracker.get_top_word_sentiments()
        sent_top = tracker.get_top_sentence_sentiments()
        msg_count, w_clusters, s_clusters, ignored = tracker.status()
        with tracker.lock:
            tracker.messages_at_last_print = tracker.message_count
        out_lines = []
        out_lines.append(f"\\n=== TOP {TOP_N} WORD CLUSTERS ===")
        if not word_top:
            out_lines.append("  (No word clusters yet)")
        else:
            for i, s in enumerate(word_top, 1):
                out_lines.append(f"{i:2}. {s['freshness']}{s['polarity']} {s['sentiment']:<16} score={s['score']:.3f} count={s['count']:>3}  ({s['polarity_label']})")
                ex = s['sentence'][:80] + ("..." if len(s['sentence']) > 80 else "")
                out_lines.append(f"     \"{ex}\"  members=[{s['members']}]")
        out_lines.append(f"\\n=== TOP {TOP_N} SIMILAR SENTENCE CLUSTERS ===")
        if not sent_top:
            out_lines.append("  (No similar sentence groups yet)")
        else:
            for i, s in enumerate(sent_top, 1):
                ex = s['sentence'][:80] + ("..." if len(s['sentence']) > 80 else "")
                out_lines.append(f"{i:2}. {s['freshness']}{s['polarity']} score={s['score']:.3f} count={s['count']:>3}  last={s['last_seen']}  ({s['polarity_label']})")
                out_lines.append(f"     \"{ex}\"")
        out_lines.append(f"\\nMessages: {msg_count} | Word clusters: {w_clusters} | Sentence clusters: {s_clusters} | Regex ignored: {ignored}")
        out_lines.append("-" * 95)
        print("\\n".join(out_lines), flush=True)

def connect_and_listen(channel, token, ignore_users, min_word_len, min_sentence_words, ignore_words,
                       ignore_mentions, ignore_commands, word_threshold, sent_threshold,
                       decay_half_life, ignore_regex):
    ignore_users_set = {u.lower().strip() for u in ignore_users} if ignore_users else set()
    ignore_words_set = {w.lower().strip() for w in ignore_words} if ignore_words else set()

    global embed_model, sentiment_pipeline
    if embed_model is None or sentiment_pipeline is None:
        raise RuntimeError("Models not loaded. Call load_models() first.")

    # Fetch third‑party emotes and add them to the ignore set
    try:
        fetched_emotes = fetch_channel_emotes(channel)
        if fetched_emotes:
            ignore_words_set.update(fetched_emotes)
    except Exception as e:
        print(f"[WARN] Failed to fetch emotes: {e}", flush=True)

    tracker = SemanticSentimentTracker(min_word_len, min_sentence_words, frozenset(ignore_words_set),
                                       float(word_threshold), float(sent_threshold),
                                       float(decay_half_life))

    # Setup queue for IRC -> consumer communication
    q = queue.Queue(maxsize=QUEUE_MAXSIZE)
    # Shared counter for dropped messages (mutable list for thread safety)
    dropped_list = [0]

    # Start IRC reader thread
    irc_thread = Thread(
        target=irc_reader_thread,
        args=(channel, token, frozenset(ignore_users_set), frozenset(ignore_words_set),
              bool(ignore_mentions), bool(ignore_commands), ignore_regex, q, dropped_list),
        daemon=True
    )
    irc_thread.start()
    print(f"[INFO] Connected to Twitch IRC for #{channel} ...", flush=True)
    print(f"[INFO] Joined #{channel}. Decoupled per-word + sentence-level semantic clustering active.", flush=True)

    # Printer as daemon thread
    stop_event = Event()
    printer = Thread(target=printer_thread, args=(tracker, stop_event), daemon=True)
    printer.start()

    # Main consumer loop: read from queue and feed tracker
    try:
        while True:
            item = q.get()
            if item is None:  # sentinel indicating end of stream
                break
            tracker.add_message(item["text"], ts=item["ts"])
    except KeyboardInterrupt:
        print("\\n[INFO] Stopping...", flush=True)
    finally:
        stop_event.set()
        printer.join(timeout=1.0)
        # Thread will stop when process exits; just report dropped stats
        dropped = dropped_list[0]
        if dropped:
            print(f"[INFO] Dropped {dropped} messages due to queue full or regex filter.", flush=True)
        print("[INFO] Disconnected.", flush=True)

def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Twitch chat sentiment analyzer")
    parser.add_argument("--channel", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--ignore-users", default="")
    parser.add_argument("--ignore-words", default="")
    parser.add_argument("--min-word-len", type=int, default=3)
    parser.add_argument("--min-sentence-words", type=int, default=2,
                        help="Minimum meaningful words for a message to contribute to word clusters (default: 2; sentence clustering always runs)")
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL, help=f"Embedding model to use (default: {DEFAULT_EMBED_MODEL})")
    parser.add_argument("--word-threshold", type=float, default=DEFAULT_WORD_SIMILARITY_THRESHOLD,
                        help=f"Similarity threshold for word clustering (default: {DEFAULT_WORD_SIMILARITY_THRESHOLD})")
    parser.add_argument("--sent-threshold", type=float, default=DEFAULT_SENTENCE_SIMILARITY_THRESHOLD,
                        help=f"Similarity threshold for sentence clustering (default: {DEFAULT_SENTENCE_SIMILARITY_THRESHOLD})")
    parser.add_argument("--decay-halflife", type=float, default=DEFAULT_DECAY_HALF_LIFE_SECONDS,
                        help=f"Half-life for score decay in seconds (default: {DEFAULT_DECAY_HALF_LIFE_SECONDS})")
    parser.add_argument("--ignore-regex", type=str, default=None,
                        help="Regex pattern to ignore matching messages (default: None)")
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

    # Convert lists
    ignore_users = [u.strip() for u in args.ignore_users.split(",") if u.strip()]
    ignore_words = [w.strip() for w in args.ignore_words.split(",") if w.strip()]

    connect_and_listen(
        args.channel.lower(), args.token, ignore_users,
        args.min_word_len, args.min_sentence_words, ignore_words,
        args.ignore_mentions, args.ignore_commands, args.word_threshold, args.sent_threshold,
        args.decay_halflife, args.ignore_regex
    )

if __name__ == "__main__":
    main()