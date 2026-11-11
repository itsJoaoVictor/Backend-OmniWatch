import asyncio
from datetime import datetime, timezone, timedelta
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete, exists, not_
from typing import List

from app.media.models import Media, MediaRelease
from app.tracking.models import UserListItem, UserEpisodeProgress
from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.tmdb_client import tmdb_get_json

async def sync_single_media_release(db: AsyncSession, media_id: str):
    """Fetches and updates release/episode data for a single media item."""
    stmt_media = select(Media).where(Media.id == media_id)
    result = await db.execute(stmt_media)
    media = result.scalars().first()
    
    if not media:
        return

    try:
        if media.media_type == "movie":
            data = await tmdb_get_json(f"/movie/{media.tmdb_id}", params={"language": "pt-BR"}, timeout=12.0)
            if data:
                rel_date_str = data.get("release_date")
                if rel_date_str:
                    media.release_date = rel_date_str
                    try:
                        r_date = datetime.strptime(rel_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                        await db.execute(delete(MediaRelease).where(MediaRelease.media_id == media.id))
                        
                        new_release = MediaRelease(
                            media_id=media.id,
                            title=media.title,
                            description=f"Lancamento do filme {media.title}",
                            release_date=r_date,
                        )
                        db.add(new_release)
                    except ValueError:
                        pass
        elif media.media_type == "tv":
            params = {"language": "pt-BR"}
            data = await tmdb_get_json(f"/tv/{media.tmdb_id}", params=params, timeout=12.0)
            if data:
                first_air_date = data.get("first_air_date")
                if first_air_date:
                    media.release_date = first_air_date
                next_episode = data.get("next_episode_to_air")
                
                await db.execute(delete(MediaRelease).where(MediaRelease.media_id == media.id))

                synced_season_episodes = False
                if next_episode and next_episode.get("season_number") is not None:
                    s_num = next_episode.get("season_number")
                    # Busca os episódios completos da temporada para registrar todos os episódios futuros confirmados
                    s_data = await tmdb_get_json(f"/tv/{media.tmdb_id}/season/{s_num}", params=params, timeout=12.0)
                    if s_data:
                        from app.core.utils import get_server_today
                        today_str = get_server_today().strftime("%Y-%m-%d")
                        for ep in s_data.get("episodes", []):
                            ep_air_date = ep.get("air_date")
                            if ep_air_date and ep_air_date >= today_str:
                                try:
                                    r_date = datetime.strptime(ep_air_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                                    e_num = ep.get("episode_number")
                                    ep_title = ep.get("name") or f"Episódio {e_num}"
                                    ep_desc = ep.get("overview", "")
                                    
                                    new_release = MediaRelease(
                                        media_id=media.id,
                                        title=f"T{s_num}E{e_num} - {ep_title}",
                                        description=ep_desc,
                                        release_date=r_date,
                                        season_number=s_num,
                                        episode_number=e_num
                                    )
                                    db.add(new_release)
                                    synced_season_episodes = True
                                except ValueError:
                                    pass

                # Fallback de segurança se a busca da temporada não trouxer episódios futuros
                if not synced_season_episodes and next_episode and next_episode.get("air_date"):
                    try:
                        r_date = datetime.strptime(next_episode.get("air_date"), "%Y-%m-%d").replace(tzinfo=timezone.utc)
                        ep_title = next_episode.get("name", "Novo Episodio")
                        ep_desc = next_episode.get("overview", "")
                        s_num = next_episode.get("season_number")
                        e_num = next_episode.get("episode_number")
                        
                        new_release = MediaRelease(
                            media_id=media.id,
                            title=f"T{s_num}E{e_num} - {ep_title}",
                            description=ep_desc,
                            release_date=r_date,
                            season_number=s_num,
                            episode_number=e_num
                        )
                        db.add(new_release)
                    except ValueError:
                        pass
    except Exception as e:
        print(f"Error syncing media {media.id}: {e}")
            
    await db.commit()

async def revert_completed_shows(db: AsyncSession):
    """Auto-revert completed TV shows to watching if a new episode has aired and is not watched."""
    now_utc = datetime.now(timezone.utc)

    # Subquery para verificar se o episódio lançado já foi assistido pelo usuário
    watched_episode_exists = exists(
        select(UserEpisodeProgress.id).where(
            UserEpisodeProgress.user_list_item_id == UserListItem.id,
            UserEpisodeProgress.season_number == MediaRelease.season_number,
            UserEpisodeProgress.episode_number == MediaRelease.episode_number
        )
    )

    from sqlalchemy.orm import joinedload
    stmt_revert = (
        select(UserListItem)
        .options(joinedload(UserListItem.media))
        .join(Media, UserListItem.media_id == Media.id)
        .join(MediaRelease, UserListItem.media_id == MediaRelease.media_id)
        .where(
            Media.media_type == "tv",
            UserListItem.status == "completed",
            MediaRelease.release_date <= now_utc,
            MediaRelease.season_number.isnot(None),
            MediaRelease.episode_number.isnot(None),
            not_(watched_episode_exists)
        )
        .distinct()
    )
    result_revert = await db.execute(stmt_revert)
    items_to_revert = result_revert.scalars().unique().all()
    
    if items_to_revert:
        from app.core.cache import tv_status_cache
        for item in items_to_revert:
            item.status = "watching"
            if item.media and item.media.tmdb_id:
                tv_status_cache.delete_prefix(f"tv_status:{item.media.tmdb_id}")
        await db.commit()

async def fetch_changed_tmdb_ids(media_type: str, start_date: str, end_date: str) -> List[int]:
    """Fetches all changed TMDB IDs for a given media type within a date range."""
    changed_ids = set()
    endpoint = f"/{media_type}/changes"
    page = 1
    total_pages = 1

    while page <= total_pages:
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "page": page
        }
        data = await tmdb_get_json(endpoint, params=params, timeout=12.0)
        if not data:
            break
        for item in data.get("results", []):
            if "id" in item:
                changed_ids.add(item["id"])
        total_pages = data.get("total_pages", 1)
        page += 1

    return list(changed_ids)

async def promote_upcoming_media(db: AsyncSession):
    """
    Promove mídias com status 'upcoming' para 'plan_to_watch' quando a data de lançamento for atingida (release_date <= hoje),
    e gera uma notificação amigável na central de notificações do usuário e via Web Push com pôster.
    """
    from app.core.utils import is_date_released
    from app.notifications.models import Notification
    from app.notifications.push_service import trigger_web_push_background
    from sqlalchemy.orm import selectinload
    
    stmt = (
        select(UserListItem)
        .join(Media, UserListItem.media_id == Media.id)
        .where(UserListItem.status == "upcoming")
        .options(selectinload(UserListItem.media))
    )
    result = await db.execute(stmt)
    upcoming_items = result.scalars().all()
    
    promoted_count = 0
    for item in upcoming_items:
        if item.media and item.media.release_date and is_date_released(item.media.release_date):
            item.status = "plan_to_watch"
            promoted_count += 1
            
            # Notifica o usuário na central in-app
            media_tipo = "O filme" if item.media.media_type == "movie" else "A série"
            notif_title = f"Estreia: {item.media.title}!"
            notif_msg = f"{media_tipo} '{item.media.title}' estreou hoje e agora está disponível na sua lista Quero Ver!"
            new_notif = Notification(
                user_id=item.user_id,
                media_id=item.media.id,
                title=notif_title,
                message=notif_msg
            )
            db.add(new_notif)

            # Envia Web Push com o pôster oficial TMDB como ícone
            poster_icon = f"https://image.tmdb.org/t/p/w185{item.media.poster_path}" if item.media.poster_path else None
            target_url = f"/{'movie' if item.media.media_type == 'movie' else 'tv'}/{item.media.tmdb_id}"
            trigger_web_push_background(
                user_id=item.user_id,
                title=f"Estreia: {item.media.title} 🎬",
                body=notif_msg,
                url=target_url,
                icon=poster_icon,
                tag=f"premiere-{item.media.id}"
            )
            
    if promoted_count > 0:
        await db.commit()
        print(f"Promoted {promoted_count} items from 'upcoming' to 'plan_to_watch'.")


async def notify_today_releases(db: AsyncSession):
    """
    Identifica lançamentos agendados para hoje (novos episódios de séries e filmes)
    para cada usuário que acompanha os títulos na sua lista (watching ou plan_to_watch).

    Regras de Anti-Spam e Concentração:
    - Agrupamento por série: se uma série lança múltiplos episódios no mesmo dia (ex: Fool Night com 12 episódios),
      gera UMA única notificação concentrada: 'Fool Night: 12 novos episódios (T1E1 a T1E12) já disponíveis!'.
    - Agrupamento do usuário: se o usuário tem até 2 séries/filmes lançando hoje, despacha 1 push individual para cada
      com o pôster oficial TMDB como ícone. Se o usuário tiver 3 ou mais títulos lançando no mesmo dia,
      condensa em um resumo diário para não inundar a barra de notificações do celular.
    - Idempotência: verifica notificações já registradas para o usuário no dia de hoje, garantindo que o push
      nunca seja enviado duas vezes para o mesmo título no mesmo dia.
    """
    from collections import defaultdict
    from app.core.utils import get_server_today
    from app.notifications.models import Notification
    from app.notifications.push_service import trigger_web_push_background
    from sqlalchemy.orm import joinedload

    today = get_server_today()
    start_today = datetime(today.year, today.month, today.day, 0, 0, 0, tzinfo=timezone.utc)
    end_today = datetime(today.year, today.month, today.day, 23, 59, 59, tzinfo=timezone.utc)

    # 1. Busca todos os MediaReleases com data de hoje
    stmt_releases = (
        select(MediaRelease)
        .options(joinedload(MediaRelease.media))
        .where(MediaRelease.release_date >= start_today, MediaRelease.release_date <= end_today)
    )
    res_releases = await db.execute(stmt_releases)
    today_releases = res_releases.scalars().all()
    if not today_releases:
        return

    # Agrupa episódios/lançamentos por media_id
    releases_by_media = defaultdict(list)
    for r in today_releases:
        releases_by_media[r.media_id].append(r)

    # Invalida cache de status das séries com estreias hoje para que a Minha Lista/Dashboard reflitam o novo episódio no ar
    from app.core.cache import tv_status_cache, details_cache
    for r in today_releases:
        if r.media and r.media.tmdb_id:
            tv_status_cache.delete_prefix(f"tv_status:{r.media.tmdb_id}")
            details_cache.delete_prefix(f"details:tv:{r.media.tmdb_id}")

    tracked_media_ids = list(releases_by_media.keys())
    if not tracked_media_ids:
        return

    # 2. Busca usuários que acompanham essas mídias (watching ou plan_to_watch)
    stmt_user_items = (
        select(UserListItem)
        .options(joinedload(UserListItem.media))
        .where(
            UserListItem.media_id.in_(tracked_media_ids),
            UserListItem.status.in_(["watching", "plan_to_watch"])
        )
    )
    res_items = await db.execute(stmt_user_items)
    user_items = res_items.scalars().all()
    if not user_items:
        return

    # Agrupa mídias por usuário
    user_media_map = defaultdict(list)
    for item in user_items:
        if item.media:
            user_media_map[item.user_id].append(item.media)

    # 3. Processa cada usuário com regras de anti-spam e idempotência
    for user_id, medias in user_media_map.items():
        # Busca mídias já notificadas hoje para esse usuário
        stmt_notified = select(Notification.media_id).where(
            Notification.user_id == user_id,
            Notification.created_at >= start_today,
            Notification.media_id.in_([m.id for m in medias])
        )
        res_notified = await db.execute(stmt_notified)
        already_notified_ids = set(res_notified.scalars().all())

        pending_medias = [m for m in medias if m.id not in already_notified_ids]
        if not pending_medias:
            continue

        # Regra Anti-Spam:
        # Até 2 títulos no dia: 1 push para cada com pôster oficial
        # 3 ou mais títulos no dia: resumo concentrado
        if len(pending_medias) <= 2:
            for media in pending_medias:
                rels = releases_by_media[media.id]
                poster_icon = f"https://image.tmdb.org/t/p/w185{media.poster_path}" if media.poster_path else None
                target_url = f"/{'movie' if media.media_type == 'movie' else 'tv'}/{media.tmdb_id}"

                if media.media_type == "movie":
                    title = f"Estreia de Hoje: {media.title} 🍿"
                    msg = f"O filme '{media.title}' estreia hoje! Confira no seu calendário."
                else:
                    if len(rels) == 1:
                        rel = rels[0]
                        ep_label = rel.title or (f"T{rel.season_number}E{rel.episode_number}" if rel.season_number else "")
                        title = f"Novo Episódio: {media.title} 🎬"
                        msg = f"O episódio {ep_label} estreou hoje! Confira no calendário."
                    else:
                        s_nums = sorted(list(set(r.season_number for r in rels if r.season_number is not None)))
                        ep_nums = [r.episode_number for r in rels if r.episode_number is not None]
                        season_prefix = f"T{s_nums[0]}" if s_nums else ""
                        if ep_nums:
                            min_ep, max_ep = min(ep_nums), max(ep_nums)
                            range_str = f"{season_prefix}E{min_ep} a {season_prefix}E{max_ep}" if min_ep != max_ep else f"{season_prefix}E{min_ep}"
                        else:
                            range_str = f"{len(rels)} episódios"

                        title = f"Novos Episódios: {media.title} 🎬"
                        msg = f"{len(rels)} novos episódios ({range_str}) acabaram de estrear hoje!"

                # Registra na central de notificações in-app
                db_notif = Notification(
                    user_id=user_id,
                    media_id=media.id,
                    title=title,
                    message=msg
                )
                db.add(db_notif)

                # Dispara Web Push
                trigger_web_push_background(
                    user_id=user_id,
                    title=title,
                    body=msg,
                    url=target_url,
                    icon=poster_icon,
                    tag=f"release-{media.id}-{today.strftime('%Y%m%d')}"
                )
        else:
            names = [m.title for m in pending_medias]
            if len(names) <= 3:
                names_str = ", ".join(names[:-1]) + " e " + names[-1]
            else:
                names_str = f"{names[0]}, {names[1]} e mais {len(names)-2} títulos"

            title = f"Lançamentos de Hoje no OmniWatch ({len(pending_medias)} títulos) 🎬"
            msg = f"{len(pending_medias)} títulos da sua lista têm novidades hoje: {names_str}. Confira no calendário!"
            first_poster = f"https://image.tmdb.org/t/p/w185{pending_medias[0].poster_path}" if pending_medias[0].poster_path else None

            # Registra no banco para cada mídia (para constar nas notificações in-app)
            for m in pending_medias:
                db.add(Notification(
                    user_id=user_id,
                    media_id=m.id,
                    title=f"Lançamento: {m.title}",
                    message="Novos episódios já disponíveis no calendário.",
                    is_read=False
                ))

            # Dispara APENAS UM Web Push concentrado para o usuário
            trigger_web_push_background(
                user_id=user_id,
                title=title,
                body=msg,
                url="/calendar",
                icon=first_poster,
                tag=f"releases-daily-{user_id}-{today.strftime('%Y%m%d')}"
            )

    await db.commit()

async def migrate_existing_future_media():
    """
    Executado no startup da aplicação:
    Garante a integridade absoluta dos status 'upcoming' e 'plan_to_watch'.
    Para cada mídia sem release_date no banco, consulta o TMDB para preencher a data real.
    - Se a data real for passada (<= hoje, ex: Corações de Ferro, Parasita, Better Call Saul), garante status 'plan_to_watch'.
    - Se a data real for futura ou se o título estiver em produção/sem data (ex: Terrifier 4), garante status 'upcoming'.
    """
    from sqlalchemy.orm import selectinload
    from app.core.utils import is_date_released
    from app.details.services import fetch_movie_details, fetch_tv_details

    try:
        async with AsyncSessionLocal() as db:
            stmt = (
                select(UserListItem)
                .join(Media, UserListItem.media_id == Media.id)
                .where(UserListItem.status.in_(["upcoming", "plan_to_watch"]))
                .options(selectinload(UserListItem.media))
            )
            res = await db.execute(stmt)
            items = res.scalars().all()
            
            reverted_to_plan_count = 0
            moved_to_upcoming_count = 0
            
            for item in items:
                media = item.media
                if not media:
                    continue

                # 1. Se a mídia não tem release_date salva no nosso banco, busca no TMDB
                if not media.release_date:
                    try:
                        if media.media_type == "movie":
                            tmdb_data = await fetch_movie_details(media.tmdb_id)
                            media.release_date = tmdb_data.release_date
                        else:
                            tmdb_data = await fetch_tv_details(media.tmdb_id)
                            media.release_date = tmdb_data.first_air_date
                    except Exception as e:
                        print(f"Erro ao buscar data TMDB para {media.title}: {e}")

                # 2. Agora com a data real verificada
                has_released = is_date_released(media.release_date)

                if has_released:
                    # Se já lançou, DEVE ser plan_to_watch!
                    if item.status == "upcoming":
                        item.status = "plan_to_watch"
                        reverted_to_plan_count += 1
                else:
                    # Se NÃO lançou (sem data ou futura), DEVE ser upcoming!
                    if item.status == "plan_to_watch":
                        item.status = "upcoming"
                        moved_to_upcoming_count += 1

            if reverted_to_plan_count > 0 or moved_to_upcoming_count > 0:
                await db.commit()
                print(f"[OmniWatch Startup] Sincronização concluída: {reverted_to_plan_count} títulos lançados restaurados para 'plan_to_watch' e {moved_to_upcoming_count} títulos futuros confirmados em 'upcoming'.")
            else:
                print("[OmniWatch Startup] Todas as mídias já estão com os status corretos.")
    except Exception as e:
        print(f"[OmniWatch Startup] Falha na sincronização de status de títulos: {e}")

async def run_calendar_sync_loop(interval_hours: int = 24, initial_delay_seconds: int = 25):
    """Background task to sync calendar data using the TMDB Changes API."""
    print("Calendar sync loop started.")
    if initial_delay_seconds > 0:
        await asyncio.sleep(initial_delay_seconds)
    while True:
        try:
            print("Running scheduled calendar sync...")
            now_utc = datetime.now(timezone.utc)
            start_date = (now_utc - timedelta(days=1)).strftime("%Y-%m-%d")
            end_date = now_utc.strftime("%Y-%m-%d")
            
            # 1. Fetch changed TMDB IDs
            changed_movies = await fetch_changed_tmdb_ids("movie", start_date, end_date)
            changed_tv = await fetch_changed_tmdb_ids("tv", start_date, end_date)
            
            all_changed_tmdb_ids = set(changed_movies + changed_tv)
            
            async with AsyncSessionLocal() as db:
                stmt_tracked_ids = select(UserListItem.media_id).where(
                    UserListItem.status.in_(["plan_to_watch", "completed", "watching", "upcoming"])
                ).distinct()
                res_tracked = await db.execute(stmt_tracked_ids)
                tracked_media_ids = res_tracked.scalars().all()
                
                if tracked_media_ids:
                    from sqlalchemy import or_
                    conditions = [(Media.release_date == None) | (Media.release_date == "")]
                    if all_changed_tmdb_ids:
                        conditions.append(Media.tmdb_id.in_(all_changed_tmdb_ids))

                    stmt_media_to_update = select(Media.id).where(
                        Media.id.in_(tracked_media_ids),
                        or_(*conditions)
                    )
                    res_media = await db.execute(stmt_media_to_update)
                    media_ids_to_update = res_media.scalars().all()
                    
                    # 3. Update each matching media
                    for m_id in media_ids_to_update:
                        await sync_single_media_release(db, m_id)
                
                # Limpa cache em memória das séries que tiveram alterações no TMDB
                if all_changed_tmdb_ids:
                    from app.core.cache import tv_status_cache, details_cache
                    for tmdb_id_changed in all_changed_tmdb_ids:
                        tv_status_cache.delete_prefix(f"tv_status:{tmdb_id_changed}")
                        details_cache.delete_prefix(f"details:tv:{tmdb_id_changed}")

                # 4. Revert completed shows if new episodes aired
                if all_changed_tmdb_ids:
                    await revert_completed_shows(db)

                # 5. Promove itens 'upcoming' que estrearam e notifica os usuários
                await promote_upcoming_media(db)

                # 6. Notifica lançamentos de hoje (episódios de séries e filmes) com anti-spam
                await notify_today_releases(db)
                    
            print("Calendar sync complete.")
        except Exception as e:
            print(f"Error in calendar sync loop: {e}")
        
        await asyncio.sleep(interval_hours * 3600)
