import os, csv, argparse, asyncio, aiohttp, re
from pathlib import Path
from typing import Optional

TMDB_API = "https://api.themoviedb.org/3"
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "../data/movielens-32m/movielens-original/ml-32m"
def parse_year_from_title(title: str) -> Optional[str]:
    match = re.search(r"\((\d{4})\)\s*$", title or "")
    return match.group(1) if match else None

async def fetch_json(session: aiohttp.ClientSession, url: str, params: dict):
    for attempt in range(3):
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status == 429:
                    wait = int(resp.headers.get("Retry-After", "2"))
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                return await resp.json()
        except Exception:
            if attempt == 2: raise
            await asyncio.sleep(1 + attempt)

async def get_tmdb_config(session, api_key):
    return await fetch_json(session, f"{TMDB_API}/configuration", {"api_key": api_key})

def build_poster_url(tmdb_config, poster_path, size="w342"):
    if not poster_path: return None
    base = (tmdb_config.get("images") or {}).get("secure_base_url", "https://image.tmdb.org/t/p/")
    sizes = (tmdb_config.get("images") or {}).get("poster_sizes", [])
    if size not in sizes: size = sizes[-2] if len(sizes) >= 2 else "w342"
    return f"{base}{size}{poster_path}"

async def fetch_movie_details(session, api_key, tmdb_id):
    return await fetch_json(session, f"{TMDB_API}/movie/{tmdb_id}", {"api_key": api_key, "language": "en-US"})

async def download_image(session, url, dest: Path, request_limit: asyncio.Semaphore):
    if not url: return
    dest.parent.mkdir(parents=True, exist_ok=True)
    async with request_limit:
        for attempt in range(3):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    resp.raise_for_status()
                    dest.write_bytes(await resp.read())
                    return
            except Exception:
                if attempt == 2: return
                await asyncio.sleep(1 + attempt)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--movielens-dir", default=str(DATA_DIR / "ml-modern"))
    parser.add_argument("--out", default=str(DATA_DIR / "ml-modern/enriched.csv"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--poster-size", default="w342")
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    api_key = os.getenv("TMDB_API_KEY")
    if not api_key: raise SystemExit("Please export TMDB_API_KEY.")

    movielens_dir = Path(args.movielens_dir)
    movies_csv = movielens_dir / "movies.csv"
    links_csv = movielens_dir / "links.csv"
    posters_dir = DATA_DIR / "posters"

    if not movies_csv.exists() or not links_csv.exists():
        raise SystemExit(f"Files not found in {movielens_dir}")

    # Load Movies
    movies_by_id = {}
    with movies_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            movies_by_id[row["movieId"]] = {
                "movieId": row["movieId"],
                "title": row["title"],
                "year": parse_year_from_title(row["title"]),
                "ml_genres": row.get("genres", "")
            }

    # Load Links (WITH FIX FOR FLOATS)
    movie_records = []
    with links_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            movie_id = row["movieId"]
            if movie_id in movies_by_id:
                raw_tmdb = row.get("tmdbId", "").strip()
                # FIX: Remove '.0' if pandas saved it as float
                if raw_tmdb.endswith(".0"):
                    raw_tmdb = raw_tmdb[:-2]
                
                if raw_tmdb.isdigit():
                    movie_records.append(movies_by_id[movie_id] | {
                        "tmdbId": raw_tmdb, 
                        "imdbId": row.get("imdbId", "").strip()
                    })

    movies_to_fetch = movie_records[: args.limit] if args.limit is not None else movie_records
    print(f"Starting scrape for {len(movies_to_fetch)} movies...")

    # Fetching Logic
    timeout = aiohttp.ClientTimeout(total=30)
    connector = aiohttp.TCPConnector(limit=0)
    limit = asyncio.Semaphore(args.concurrency)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tmdb_config = await get_tmdb_config(session, api_key)
        
        async def fetch_task(entry):
            try:
                async with limit:
                    data = await fetch_movie_details(session, api_key, entry["tmdbId"])
            except Exception as e: 
                # Silent fail for 404s to keep logs clean
                return None
            
            poster = build_poster_url(tmdb_config, data.get("poster_path"), args.poster_size)
            tmdb_genres = [g["name"] for g in (data.get("genres") or [])]
            
            return ({
                **entry,
                "tmdb_title": data.get("title"),
                "tmdb_overview": (data.get("overview") or "").replace("\n", " ").strip(),
                "tmdb_genres": "|".join(tmdb_genres),
                "poster_url": poster
            }, poster)

        results = await asyncio.gather(*(fetch_task(m) for m in movies_to_fetch))
        valid_results = [r for r in results if r]

        # Download Images
        if args.download_images:
            print(f"Downloading {len(valid_results)} images...")
            tasks = []
            for item, url in valid_results:
                if url:
                    tasks.append(download_image(session, url, posters_dir / f"{item['movieId']}.jpg", limit))
            await asyncio.gather(*tasks)

    # Save CSV
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    keys = ["movieId","tmdbId","imdbId","ml_title","ml_year","ml_genres",
            "tmdb_title","tmdb_overview","tmdb_genres","poster_url"]
    
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        for item, _ in valid_results:
            writer.writerow(item)

    print(f"Done. Saved to {args.out}")

if __name__ == "__main__":
    asyncio.run(main())