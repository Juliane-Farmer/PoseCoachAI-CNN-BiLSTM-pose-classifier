import os
import re
import csv
import json
import time
import argparse
from pathlib import Path
from urllib.parse import urlencode
import requests
import yt_dlp

YOUTUBE_API = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS = "https://www.googleapis.com/youtube/v3/videos"

EX_DEFAULT = ["Squats","Push Ups","Pull ups","Jumping Jacks","Russian twists"]

def slug(s):
    s = re.sub(r"[^\w\-]+", "-", s.strip().lower())
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "untitled"

def fname_safe(s):
    return re.sub(r'[\\/*?:"<>|]+', "", s)

def yt_search(api_key, query, max_results=15, page_token=None):
    params = {
        "key": api_key,
        "q": query,
        "type": "video",
        "part": "id",
        "videoLicense": "creativeCommon",
        "videoEmbeddable": "true",
        "safeSearch": "none",
        "maxResults": max_results,
        "pageToken": page_token or "", }
    r = requests.get(YOUTUBE_API, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def yt_videos(api_key, ids):
    params = {
        "key": api_key,
        "id": ",".join(ids),
        "part": "snippet,contentDetails,statistics,status", }
    r = requests.get(YOUTUBE_VIDEOS, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def iso8601_duration_to_seconds(s):
    pat = r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?"
    m = re.match(pat, s or "")
    if not m:
        return 0
    h = int(m.group(1) or 0)
    m_ = int(m.group(2) or 0)
    s_ = int(m.group(3) or 0)
    return h*3600 + m_*60 + s_

def collect_cc_videos(api_key, query, want=8, max_pages=5, accept_long=False):
    ids = []
    token = None
    pages = 0
    while len(ids) < want and pages < max_pages:
        js = yt_search(api_key, query, max_results=min(50, want*2), page_token=token)
        ids.extend([it["id"]["videoId"] for it in js.get("items", []) if it.get("id", {}).get("videoId")])
        token = js.get("nextPageToken")
        pages += 1
        if not token:
            break
        time.sleep(0.3)
    if not ids:
        return []
    meta = yt_videos(api_key, ids[:50]).get("items", [])
    out = []
    for it in meta:
        vid = it.get("id")
        sn = it.get("snippet", {})
        cd = it.get("contentDetails", {})
        st = it.get("status", {})
        if not vid or st.get("uploadStatus") != "processed":
            continue
        dur = iso8601_duration_to_seconds(cd.get("duration"))
        if dur == 0:
            continue
        short = dur < 4*60
        medium = 4*60 <= dur <= 20*60
        if not (short or medium or accept_long):
            continue
        title = sn.get("title") or ""
        ch = sn.get("channelTitle") or ""
        out.append({
            "id": vid,
            "title": title,
            "uploader": ch,
            "duration_sec": dur,
            "license": "Creative Commons",
            "url": f"https://www.youtube.com/watch?v={vid}", })
    out.sort(key=lambda x: x["duration_sec"])
    return out[:want]

def ensure_dirs():
    Path("dataset/videos").mkdir(parents=True, exist_ok=True)
    Path("dataset").mkdir(parents=True, exist_ok=True)

def build_outtmpl():
    return "dataset/videos/%(webpage_url_domain)s-%(uploader)s-%(title).120s-%(id)s.%(ext)s"

def download_batch(urls):
    ydl_opts = {
        "outtmpl": build_outtmpl(),
        "merge_output_format": "mp4",
        "format": "mp4[height<=720][vcodec*=avc1]+bestaudio/best/best",
        "quiet": False,
        "noprogress": False,
        "ignoreerrors": True,
        "retries": 3,
        "concurrent_fragment_downloads": 3,
        "trim_file_name": 180,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(urls)

def main():
    ap = argparse.ArgumentParser(description="Fetch Creative Commons exercise videos and save to dataset/videos/")
    ap.add_argument("--exercises", type=str, default=",".join(EX_DEFAULT))
    ap.add_argument("--per-class", type=int, default=6)
    ap.add_argument("--query-extra", type=str, default="exercise tutorial form")
    ap.add_argument("--accept-long", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    api_key = os.getenv("YT_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing YT_API_KEY environment variable")

    ensure_dirs()
    manifest_rows = []
    wanted = [e.strip() for e in args.exercises.split(",") if e.strip()]
    all_urls = []
    for ex in wanted:
        q = f'{ex} {args.query_extra}'.strip()
        vids = collect_cc_videos(api_key, q, want=args.per_class, accept_long=args.accept_long)
        for v in vids:
            title_tag = fname_safe(v["title"])
            uploader_tag = slug(v["uploader"]) or "uploader"
            ex_tag = slug(ex)
            base = f"{ex_tag}--{uploader_tag}-{title_tag}-{v['id']}.mp4"
            target = Path("dataset/videos") / base
            v["target"] = str(target)
            v["exercise"] = ex
            manifest_rows.append(v)
            all_urls.append(v["url"])
    if args.dry_run:
        print(json.dumps(manifest_rows, indent=2))
        return
    if all_urls:
        download_batch(all_urls)
    for row in manifest_rows:
        if not Path(row["target"]).exists():
            globbed = list(Path("dataset/videos").glob(f"*{row['id']}.*"))
            if globbed:
                ext = globbed[0].suffix
                try:
                    Path(globbed[0]).rename(row["target"])
                except Exception:
                    pass
    meta_csv = Path("dataset/video_sources.csv")
    header = ["id","url","title","uploader","license","duration_sec","exercise","target"]
    write_header = not meta_csv.exists()
    with open(meta_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        for r in manifest_rows:
            w.writerow({k: r.get(k) for k in header})
    print(f"Saved {len(manifest_rows)} records to {meta_csv}")
    print("Done.")

if __name__ == "__main__":
    main()
