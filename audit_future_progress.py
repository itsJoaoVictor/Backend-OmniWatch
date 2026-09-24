"""
Script de Saneamento de Mídias e Episódios Futuros
OmniWatch - Conexão via asyncpg no PostgreSQL (127.0.0.1:5433)
"""

import asyncio
import os
from datetime import datetime, timezone
import httpx
from dotenv import load_dotenv
import asyncpg

load_dotenv()

DB_USER = os.getenv("DB_USER", "omniwatch")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Jvgf1211")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "5433"))
DB_NAME = os.getenv("DB_NAME", "omniwatch")
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")

def is_released(date_str: str | None) -> bool:
    if not date_str:
        return False
    try:
        rel_d = datetime.strptime(date_str.strip()[:10], "%Y-%m-%d").date()
        today_d = datetime.now(timezone.utc).date()
        return rel_d <= today_d
    except Exception:
        return False

async def fetch_tv_seasons_data(tmdb_id: int):
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_API_KEY}"
    }
    async with httpx.AsyncClient() as client:
        res = await client.get(f"https://api.themoviedb.org/3/tv/{tmdb_id}?language=pt-BR", headers=headers, timeout=10.0)
        if res.status_code != 200:
            return None, {}
        tv_data = res.json()
        series_status = tv_data.get("status", "")
        
        episodes_map = {}
        for s in tv_data.get("seasons", []):
            s_num = s.get("season_number", 0)
            if s_num > 0:
                res_s = await client.get(f"https://api.themoviedb.org/3/tv/{tmdb_id}/season/{s_num}?language=pt-BR", headers=headers, timeout=10.0)
                if res_s.status_code == 200:
                    s_data = res_s.json()
                    for ep in s_data.get("episodes", []):
                        ep_num = ep.get("episode_number")
                        episodes_map[(s_num, ep_num)] = ep.get("air_date")
        return series_status, episodes_map

async def fetch_movie_release_date(tmdb_id: int):
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_API_KEY}"
    }
    async with httpx.AsyncClient() as client:
        res = await client.get(f"https://api.themoviedb.org/3/movie/{tmdb_id}?language=pt-BR", headers=headers, timeout=10.0)
        if res.status_code != 200:
            return None
        return res.json().get("release_date")

async def main():
    print("=" * 80)
    print("  OmniWatch - Saneamento de Episódios e Filmes Futuros")
    print(f"  Data de Referência (Hoje UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d')}")
    print(f"  Conectando a {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}...")
    print("=" * 80)

    conn = await asyncpg.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME
    )
    print("Conexão com PostgreSQL estabelecida com sucesso!\n")

    # 1. Auditoria e Saneamento de Séries
    tv_items = await conn.fetch("""
        SELECT u.id as user_id, u.name as user_name,
               uli.id as item_id, uli.status as item_status,
               m.id as media_id, m.title, m.tmdb_id, m.media_type
        FROM user_list_items uli
        JOIN users u ON u.id = uli.user_id
        JOIN media m ON m.id = uli.media_id
        WHERE m.media_type = 'tv';
    """)

    print(f"Verificando {len(tv_items)} séries na lista dos usuários...")
    deleted_episodes = 0
    updated_series = 0

    for item in tv_items:
        progresses = await conn.fetch("""
            SELECT id, season_number, episode_number
            FROM user_episode_progress
            WHERE user_list_item_id = $1
            ORDER BY season_number, episode_number;
        """, item["item_id"])

        series_status, episodes_map = await fetch_tv_seasons_data(item["tmdb_id"])
        if not episodes_map:
            continue

        future_prog_ids = []
        for p in progresses:
            air_date = episodes_map.get((p["season_number"], p["episode_number"]))
            if not is_released(air_date):
                future_prog_ids.append((p["id"], p["season_number"], p["episode_number"], air_date))

        if future_prog_ids:
            print(f"\n[Série] '{item['title']}' do usuário '{item['user_name']}':")
            for pid, s_num, e_num, ad in future_prog_ids:
                print(f"  -> Removendo episódio não lançado S{s_num} E{e_num} (Estreia: {ad or 'Indefinida'})...")
                await conn.execute("DELETE FROM user_episode_progress WHERE id = $1;", pid)
                deleted_episodes += 1

        has_future_in_series = any(not is_released(ad) for ad in episodes_map.values())
        if item["item_status"] == "completed" and has_future_in_series:
            await conn.execute("UPDATE user_list_items SET status = 'watching' WHERE id = $1;", item["item_id"])
            print(f"  -> Status de '{item['title']}' ajustado de 'completed' para 'watching' (Em dia).")
            updated_series += 1

    # 2. Auditoria e Saneamento de Filmes Futuros
    movie_items = await conn.fetch("""
        SELECT u.id as user_id, u.name as user_name,
               uli.id as item_id, uli.status as item_status,
               m.id as media_id, m.title, m.tmdb_id, m.release_date
        FROM user_list_items uli
        JOIN users u ON u.id = uli.user_id
        JOIN media m ON m.id = uli.media_id
        WHERE m.media_type = 'movie' AND uli.status = 'completed';
    """)

    print(f"\nVerificando {len(movie_items)} filmes marcados como assistidos...")
    updated_movies = 0

    for m in movie_items:
        rel_date = m["release_date"]
        if not rel_date:
            rel_date = await fetch_movie_release_date(m["tmdb_id"])
        if not is_released(rel_date):
            print(f"\n[Filme] '{m['title']}' do usuário '{m['user_name']}' tem data de estreia futura ({rel_date}).")
            await conn.execute("UPDATE user_list_items SET status = 'plan_to_watch' WHERE id = $1;", m["item_id"])
            print(f"  -> Status ajustado de 'completed' para 'plan_to_watch'.")
            updated_movies += 1

    print("\n" + "=" * 80)
    print("  RESUMO DO SANEAMENTO EXECUTADO")
    print(f"  - Episódios futuros deletados: {deleted_episodes}")
    print(f"  - Séries em andamento ajustadas para 'watching': {updated_series}")
    print(f"  - Filmes não lançados ajustados para 'plan_to_watch': {updated_movies}")
    print("=" * 80)

    await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
