import os, csv, argparse, asyncio, aiohttp, re
from pathlib import Path

TMDB_API = "https://api.themoviedb.org/3"
BASE_DIR = Path(__file__).resolve().parents[1]  # Goes to project root
DATA_DIR = BASE_DIR / "movielens-data"

def parse_year_from_title(title: str) -> str | None:
    """Extract the release year (YYYY) from titles like 'Toy Story (1995)'."""
    match = re.search(r"\((\d{4})\)\s*$", title or "")
    return match.group(1) if match else None


async def fetch_json(session: aiohttp.ClientSession, url: str, params: dict):
    for attempt in range(3):
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status == 429:  # TMDB rate-limit
                    wait = int(resp.headers.get("Retry-After", "2"))
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                return await resp.json()
        except Exception:
            if attempt == 2:
                raise
            await asyncio.sleep(1 + attempt)


async def get_tmdb_config(session, api_key):
    return await fetch_json(session, f"{TMDB_API}/configuration", {"api_key": api_key})


def build_poster_url(tmdb_config, poster_path, size="w342"):
    if not poster_path:
        return None
    base = (tmdb_config.get("images") or {}).get("secure_base_url", "https://image.tmdb.org/t/p/")
    sizes = (tmdb_config.get("images") or {}).get("poster_sizes", [])
    if size not in sizes:
        size = sizes[-2] if len(sizes) >= 2 else "w342"
    return f"{base}{size}{poster_path}"


async def fetch_movie_details(session, api_key, tmdb_id):
    return await fetch_json(session, f"{TMDB_API}/movie/{tmdb_id}", {
        "api_key": api_key,
        "language": "en-US"
    })


async def download_image(session, url, dest: Path, request_limit: asyncio.Semaphore):
    if not url:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    async with request_limit:
        for attempt in range(3):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    resp.raise_for_status()
                    data = await resp.read()
                    dest.write_bytes(data)
                    return
            except Exception:
                if attempt == 2:
                    return
                await asyncio.sleep(1 + attempt)


async def main():
    parser = argparse.ArgumentParser(description="Enrich MovieLens data with TMDB metadata.")
    parser.add_argument("--movielens-dir", default=str(DATA_DIR / "ml-32m"), help="Directory containing MovieLens CSV files.")
    parser.add_argument("--out", default=str(DATA_DIR / "ml32m_tmdb_enriched.csv"), help="Output CSV file path.")
    parser.add_argument("--limit", type=int, help="Number of movies to process (use for testing!).")
    parser.add_argument("--download-images", action="store_true", help=f"Download poster images to {DATA_DIR / 'posters'}.")
    parser.add_argument("--poster-size", default="w342", help="TMDB poster size (e.g., w185, w342, w500, original).")
    parser.add_argument("--concurrency", type=int, default=20, help="Maximum concurrent TMDB requests.")
    args = parser.parse_args()

    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        raise SystemExit("Please export your TMDB_API_KEY environment variable (Create one from TMDB).")

    movielens_dir = Path(args.movielens_dir)
    movies_csv = movielens_dir / "movies.csv"
    links_csv = movielens_dir / "links.csv"
    posters_dir = DATA_DIR / "posters"

    if not movies_csv.exists() or not links_csv.exists():
        raise SystemExit("movies.csv or links.csv not found in the directory you specified.")

    # --- Load MovieLens movie data from CSV ---
    movies_by_id = {}
    with movies_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            movies_by_id[row["movieId"]] = {
                "movieId": row["movieId"],
                "title": row["title"],
                "year": parse_year_from_title(row["title"]),
                "ml_genres": row.get("genres", "")
            }

    # --- Combine MovieLens data with TMDB/IMDB IDs ---
    movie_records = []
    with links_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            movie_id = row["movieId"]
            if movie_id in movies_by_id:
                tmdb_id = row.get("tmdbId", "").strip()
                imdb_id = row.get("imdbId", "").strip()
                if tmdb_id.isdigit():
                    merged_entry = movies_by_id[movie_id] | {"tmdbId": tmdb_id, "imdbId": imdb_id}
                    movie_records.append(merged_entry)

    movies_to_fetch = movie_records[: args.limit] if args.limit is not None else movie_records

    timeout = aiohttp.ClientTimeout(total=30)
    connector = aiohttp.TCPConnector(limit=0)
    request_limit = asyncio.Semaphore(args.concurrency)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tmdb_config = await get_tmdb_config(session, api_key)

        async def fetch_and_merge_movie(movie_entry):
            """Fetch TMDB data for one movie; skip gracefully if TMDB ID not found (404)."""
            try:
                async with request_limit:
                    tmdb_data = await fetch_movie_details(session, api_key, movie_entry["tmdbId"])
            except aiohttp.ClientResponseError as e:
                if e.status == 404:
                    print(f"[SKIP] TMDB 404 for id={movie_entry['tmdbId']} ({movie_entry['title']})")
                    return None
                elif e.status in (401, 403):
                    raise SystemExit("TMDB auth error (401/403). Check your TMDB_API_KEY.") from e
                else:
                    raise

            tmdb_genres = [g["name"] for g in (tmdb_data.get("genres") or []) if g.get("name")]
            poster_url = build_poster_url(tmdb_config, tmdb_data.get("poster_path"), size=args.poster_size)
            enriched_movie = {
                "movieId": movie_entry["movieId"],
                "tmdbId": movie_entry["tmdbId"],
                "imdbId": tmdb_data.get("imdb_id") or movie_entry.get("imdbId"),
                "ml_title": movie_entry["title"],
                "ml_year": movie_entry["year"],
                "ml_genres": movie_entry["ml_genres"],
                "tmdb_title": tmdb_data.get("title"),
                "original_title": tmdb_data.get("original_title"),
                "release_date": tmdb_data.get("release_date"),
                "tmdb_overview": (tmdb_data.get("overview") or "").replace("\n", " ").strip(),
                "tmdb_genres": "|".join(tmdb_genres),
                "tmdb_vote_average": tmdb_data.get("vote_average"),
                "tmdb_vote_count": tmdb_data.get("vote_count"),
                "poster_url": poster_url,
            }
            return enriched_movie, poster_url

        # Gather results concurrently
        results = await asyncio.gather(*(fetch_and_merge_movie(m) for m in movies_to_fetch))
        results = [r for r in results if r is not None]  # drop skipped movies

        # --- Optional: download posters ---
        if args.download_images:
            image_download_tasks = []
            for enriched_movie, poster_url in results:
                if poster_url:
                    filename = f"{enriched_movie['movieId']}_{enriched_movie['tmdbId']}.jpg"
                    destination = posters_dir / filename
                    image_download_tasks.append(download_image(session, poster_url, destination, request_limit))
            if image_download_tasks:
                await asyncio.gather(*image_download_tasks)

    # --- Write CSV output ---
    fieldnames = [
        "movieId","tmdbId","imdbId","ml_title","ml_year","ml_genres",
        "tmdb_title","original_title","release_date",
        "tmdb_overview","tmdb_genres","tmdb_vote_average","tmdb_vote_count","poster_url"
    ]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for enriched_movie, _ in results:
            writer.writerow(enriched_movie)

    print(f"Done. Wrote CSV to: {args.out}")
    if args.download_images:
        print(f"Poster images saved under: {posters_dir}")


if __name__ == "__main__":
    asyncio.run(main())