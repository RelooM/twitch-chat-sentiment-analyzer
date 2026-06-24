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
  - **Toggle @mentions** (`--ignore-mentions` / `--no-ignore-mentions`)
  - **Toggle chat commands** (`--ignore-commands` / `--no-ignore-commands`)
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

| Option                    | Default | Description                                      |
|---------------------------|---------|--------------------------------------------------|
| `--channel`               | —       | Twitch channel to monitor (required)             |
| `--token`                 | —       | Twitch OAuth token (required)                    |
| `--ignore-users`          | —       | Comma-separated list of users to ignore          |
| `--ignore-words`          | —       | Comma-separated list of words to ignore          |
| `--min-word-len`          | 3       | Minimum characters per word                      |
| `--min-sentence-words`    | 2       | Minimum number of words in a message             |
| `--model`                 | all-MiniLM-L6-v2 | Embedding model for sentence-transformers   |
| `--word-threshold`        | 0.72    | Similarity threshold for word clustering         |
| `--sent-threshold`        | 0.78    | Similarity threshold for sentence clustering     |
| `--ignore-mentions`       | True    | Ignore words starting with @                     |
| `--no-ignore-mentions`    | —       | Disable @mention filtering                       |
| `--ignore-commands`       | True    | Ignore messages starting with `!`                |
| `--no-ignore-commands`    | —       | Disable command filtering                        |

## Output

The tool displays two live-updating sections:

### Word Clusters
Top semantically grouped words with representative sentences.

### Similar Sentence Clusters
Groups of nearly identical or highly similar chat messages (e.g. repeated hype, complaints, etc.).

## How It Works

1. Connects to Twitch IRC
2. Optionally filters @mentions and !commands
3. Tokenizes messages with configurable minimum length
4. Runs sentiment analysis on individual words
5. Uses `sentence-transformers` to cluster both words and full sentences
6. Applies exponential decay so recent chat ranks higher

## License

MIT License

## Author

Created by RelooM

---

*This tool is intended for entertainment, research, and moderation assistance purposes.*