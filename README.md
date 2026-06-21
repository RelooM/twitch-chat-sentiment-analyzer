# Twitch Chat Sentiment Analyzer

Real-time sentiment analysis tool for Twitch chat with **per-word** and **sentence-level semantic clustering**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Per-word semantic analysis** — Every meaningful word is analyzed individually
- **Sentence-level clustering** — Groups similar chat messages together using embeddings
- **Freshness-weighted scoring** — Newer messages have higher influence
- **Smart filtering**
  - Ignore specific users (`--ignore-users`)
  - Ignore specific words (`--ignore-words`)
  - Minimum word length (`--min-word-len`)
  - Minimum sentence length (`--min-sentence-words`)
- **Ignores @mentions** automatically
- **Deduplicates repeated words** within the same message

## Installation

### Requirements

- Python 3.10+
- `sentence-transformers`
- `transformers`
- `torch` (CPU or GPU)

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

### Advanced Usage

```bash
python3 twitch_sentiment_tool.py \
  --channel montanablack88 \
  --token oauth:xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
  --ignore-users bot1,bot2,streamlabs \
  --ignore-words pog,gg,ez,lol,wtf \
  --min-word-len 4 \
  --min-sentence-words 3
```

## Command Line Options

| Option                    | Default | Description                                      |
|---------------------------|---------|--------------------------------------------------|
| `--channel`               | —       | Twitch channel to monitor (required)             |
| `--token`                 | —       | Twitch OAuth token (required)                    |
| `--ignore-users`          | —       | Comma-separated list of users to ignore          |
| `--ignore-words`          | —       | Comma-separated list of words to ignore          |
| `--min-word-len`          | 3       | Minimum characters per word                      |
| `--min-sentence-words`    | 2       | Minimum number of words in a message             |

## Output

The tool displays two live-updating sections:

### Word Clusters
Shows the top semantically similar words and their representative sentences.

### Similar Sentence Clusters
Groups nearly identical or highly similar chat messages together (e.g., repeated hype messages, complaints about lag, etc.).

Both views use **freshness-weighted scoring** so recent chat has more influence.

## How It Works

1. Connects to Twitch IRC
2. Tokenizes every message
3. Runs sentiment analysis on individual words
4. Uses `sentence-transformers` embeddings to cluster:
   - Similar words
   - Similar full sentences
5. Applies exponential decay (45s half-life) so newer messages rank higher

## License

MIT License

## Author

Created by RelooM

---

*This tool is intended for entertainment, research, and moderation assistance purposes.*