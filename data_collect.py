import os
import pandas as pd
from datetime import datetime

# Option 1: snscrape (no API key required)
def fetch_with_snscrape(query: str, limit: int = 1000):
    """
    Returns DataFrame with columns: id, date, username, content, url
    Example query: "iPhone since:2025-09-01 until:2025-09-20 lang:en"
    """
    try:
        import snscrape.modules.twitter as sntwitter
    except Exception as e:
        raise RuntimeError("snscrape not installed or import failed") from e

    rows = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i >= limit:
            break
        rows.append({
            "id": tweet.id,
            "date": tweet.date,
            "username": tweet.user.username if tweet.user else None,
            "content": tweet.content,
            "url": f"https://twitter.com/{tweet.user.username}/status/{tweet.id}" if tweet.user else None
        })
    return pd.DataFrame(rows)

# Option 2: Tweepy (Twitter API v2)
def fetch_with_tweepy(bearer_token: str, query: str, max_results=100):
    """
    Requires Twitter API v2 bearer token.
    Returns DataFrame (id, date, username, content).
    """
    import tweepy
    client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
    tweets = []
    paginator = tweepy.Paginator(
        client.search_recent_tweets,
        query=query,
        tweet_fields=["created_at","lang","author_id","text"],
        expansions=["author_id"],
        max_results=100
    )
    users_map = {}
    for page in paginator:
        if not page.data:
            continue
        # build author map
        if page.includes and 'users' in page.includes:
            for u in page.includes['users']:
                users_map[u.id] = u.username
        for t in page.data:
            tweets.append({
                "id": t.id,
                "date": t.created_at,
                "username": users_map.get(t.author_id),
                "content": t.text
            })
            if len(tweets) >= max_results:
                break
        if len(tweets) >= max_results:
            break
    import pandas as pd
    return pd.DataFrame(tweets)
