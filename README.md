# Twitch Chat Sentiment Analyzer

Real-time sentiment analysis tool for Twitch chat with **per-word** and **sentence-level semantic clustering**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Guaranteed message counting** — Every non-command message is counted and tracked
- **Per-word semantic clustering** — Meaningful words are grouped by semantic similarity + sentiment
- **Sentence-level clustering** — Similar chat messages grouped together using embeddings
- **Variable refresh display** — Prints every 2-5 seconds, adaptive to chat activity level
- **Freshness-weighted scoring** — Newer messages have higher influence; score decays over time
- **Decoupled IRC streaming** — Chat reading runs in a separate thread with bounded queue, never blocked by analysis
- **Smart filtering**
  - Ignore specific users (`--ignore-users`)
  - Ignore specific words (`--ignore-words`)
  - Minimum word length (`--min-word-len`)
  - Minimum sentence length (`--min-sentence-words`)
  - **Toggle @mentions** (`--ignore-mentions` / `--no-ignore-mentions`)
  - **Toggle chat commands** (`--ignore-commands` / `--no-ignore-commands`)
- **Automatic third‑party emote filtering** — Fetches and ignores BTTV, FFZ, and 7TV emotes for the channel (no extra flag needed)
- **Configurable embedding models** — Choose from various sentence-transformers models
- **Adjustable similarity thresholds** — Fine-tune clustering sensitivity
- Deduplicates repeated words within the same message

## Installation

### Requirements

- Python 3.10+
- `sentence-transformers`
- `transformers`
- `torch`

### Quick Start

```bash
git clone https://github.com/RelooM/twitch-chat-sentiment-analyzer.git
cd twitch-chat-sentiment-analyzer

pip install sentence-transformers transformers torch
```

## Usage

You need a Twitch OAuth token with the `chat:read` scope.

Get one here: [https://twitchapps.com/tmi](https://twitchapps.com/tmi)

### Basic Usage

```bash
python3 twitch_sentiment_tool.py \
  --channel yourchannel \
  --token oauth:xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Advanced Usage with Toggles

```bash
python3 twitch_sentiment_tool.py \
  --channel montanablack88 \
  --token oauth:xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
  --ignore-users bot1,bot2 \
  --ignore-words pog,gg,ez \
  --min-word-len 4 \
  --min-sentence-words 3 \
  --no-ignore-mentions \
  --ignore-commands
```

### Custom Model and Thresholds

```bash
python3 twitch_sentiment_tool.py \
  --channel yourchannel \
  --token oauth:xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
  --model all-MiniLM-L12-v2 \
  --word-threshold 0.75 \
  --sent-threshold 0.80
```

## Command Line Options

|| Option                    | Default | Description                                      ||
||---------------------------|---------|--------------------------------------------------||
|| `--channel`               | —       | Twitch channel to monitor (required)             ||
|| `--token`                 | —       | Twitch OAuth token (required)                    ||
|| `--ignore-users`          | —       | Comma-separated list of users to ignore          ||
|| `--ignore-words`          | —       | Comma-separated list of words to ignore          ||
|| `--min-word-len`          | 3       | Minimum characters per word                      ||
|| `--min-sentence-words`    | 2       | Minimum number of words in a message             ||
|| `--model`                 | all-MiniLM-L6-v2 | Embedding model for sentence-transformers   ||
|| `--word-threshold`        | 0.72    | Similarity threshold for word clustering         ||
|| `--sent-threshold`        | 0.78    | Similarity threshold for sentence clustering     ||
|| `--ignore-mentions`       | True    | Ignore words starting with @                     ||
|| `--no-ignore-mentions`    | —       | Disable @mention filtering                       ||
|| `--ignore-commands`       | True    | Ignore messages starting with `!`                ||
|| `--no-ignore-commands`    | —       | Disable command filtering                        ||

## Output

The tool displays two live-updating sections (refresh every 2–5 seconds, adaptive to activity):

### Word Clusters
Top semantically grouped words with sentiment polarity, freshness indicators, and representative sentences.
- **➕** = positive sentiment · **➖** = negative sentiment · **⚪** = neutral
- **🆕** = seen in the last 30 seconds; **🔄** = recurring cluster (seen before but still active)
- Score decays exponentially — recent chat ranks higher
- Polarity tracked as net score: messages with `score > 0.6` shift polarity; mixed clusters show both

### Similar Sentence Clusters
Groups of nearly identical or highly similar chat messages (e.g. repeated hype, complaints, etc.), with polarity labels and cumulative score.

### Example Output
```
 1. 🆕➖ considering      score=10.597 count= 15  (neg)
     "@minion_laughing_guy we gotta ponder the REAL things yk? Considering"  members=[considering]
 2. 🆕➕ holy             score=1.316 count=  2  (pos)
     "HOLY ONER Susge"  members=[holy]
 3. 🆕➖ bard             score=1.578 count=  3  (neg)
     "Considering BARD IS INTING I JUST CANT PROVE IT Considering"  members=[bard]

Messages: 18 | Word clusters: 47 | Sentence clusters: 17
```

## How It Works

1. **Producer thread** connects to Twitch IRC, fetches third‑party emotes (BTTV, FFZ, 7TV) for the channel, and streams raw messages into a thread‑safe queue — never blocks on analysis.
2. **Consumer (main thread)** reads from the queue, runs sentiment + clustering at its own pace.
3. **Printer thread** reads the tracker every 2-5 seconds (faster during active chat, slower when quiet).
4. Messages are tokenized with configurable minimum length, optionally filtered for @mentions and !commands.
5. Stopwords, user‑ignored words, word‑ignored list, and the fetched emote set are removed before sentiment analysis.
6. Sentiment analysis runs on the cleaned text using a `distilbert-base-uncased-finetuned-sst-2-english` model.
7. Embeddings via `sentence-transformers` cluster both words (top‑N most distinctive) and full sentences.
8. Score decays exponentially (configurable half‑life) so recent chat ranks higher; polarity flips when score crosses 0.6.

## License

MIT License

## Author

Created by RelooM

---

*This tool is intended for entertainment, research, and moderation assistance purposes.*