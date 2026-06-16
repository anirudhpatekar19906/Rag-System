"""
youtube_loader.py
-----------------
Fetches the transcript of a YouTube video using youtube-transcript-api.

Returns a list of dicts:
    [{"start": 12.4, "text": "...", "source": "yt:<video_id>@12s"}, ...]

Supports plain video IDs or full YouTube URLs.
"""

import re
from youtube_transcript_api import YouTubeTranscriptApi

def _extract_video_id(url_or_id: str) -> str:
    """
    Parse a YouTube URL or raw video ID and return the 11-char video ID.

    Handles formats:
        https://www.youtube.com/watch?v=dQw4w9WgXcQ
        https://youtu.be/dQw4w9WgXcQ
        dQw4w9WgXcQ
    """
    patterns = [
        r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})",
    ]
    for pat in patterns:
        m = re.search(pat, url_or_id)
        if m:
            return m.group(1)
    # Assume the input IS the video ID if no pattern matched
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id
    raise ValueError(f"Cannot extract video ID from: {url_or_id!r}")


def load_youtube(url_or_id: str, languages: list[str] | None = None) -> list[dict]:
    """
    Fetch a YouTube transcript and return it as a list of segment dicts.

    Args:
        url_or_id:  Full YouTube URL or bare video ID.
        languages:  Preferred transcript languages, e.g. ["en", "en-US"].
                    Falls back to whatever is available if None.

    Returns:
        List of {"start": float, "duration": float, "text": str, "source": str}.
    """
    video_id = _extract_video_id(url_or_id)
    langs = languages or ["en", "en-US", "en-GB"]

    # Instantiate the API client for this version of the library
    api = YouTubeTranscriptApi()

    try:
        # Use .fetch() instead of .get_transcript()
        segments = api.fetch(video_id, languages=langs)
    except Exception:
        # Try any available language as a fallback
        try:
            transcript_list = api.list(video_id)
            # Find the first available transcript
            transcript = transcript_list.find_transcript(
                [t.language_code for t in transcript_list]
            )
            segments = transcript.fetch()
        except Exception as e:
            print(f"[youtube_loader] Failed to retrieve transcript for {video_id}: {e}")
            return []

    result = []
    for seg in segments:
        start = seg.start
        result.append({
            "start": start,
            "duration": seg.duration if hasattr(seg, 'duration') else 0.0,
            "text": seg.text.strip(),
            "source": f"yt:{video_id}@{int(start)}s",
        })

    print(f"[youtube_loader] Loaded {len(result)} segments from video '{video_id}'")
    return result


def load_youtube_as_text(url_or_id: str, languages: list[str] | None = None) -> str:
    """
    Convenience wrapper — returns the full transcript as a single string.
    """
    segments = load_youtube(url_or_id, languages)
    return " ".join(s["text"] for s in segments)


# ── quick smoke-test (requires a valid API key) ───────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python youtube_loader.py <youtube_url_or_id>")
        sys.exit(1)

    segs = load_youtube(sys.argv[1])
    for s in segs[:5]:
        print(f"[{s['start']:.1f}s] {s['text']}")
