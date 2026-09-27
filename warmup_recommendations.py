import sys
import os
sys.path.insert(0, os.path.abspath("."))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import asyncio
from sqlalchemy.future import select
from app.core.database import AsyncSessionLocal
from app.users.models import User
from app.recommendation.service import (
    get_user_recommendations_record,
    compute_and_save_user_recommendations,
)

async def warmup_all_users(force: bool = False):
    print("=" * 60)
    print("🚀 INICIANDO PRÉ-AQUECIMENTO DE RECOMENDAÇÕES PARA TODOS OS USUÁRIOS")
    print("=" * 60)

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User))
        users = result.scalars().all()
        print(f"Total de usuários encontrados: {len(users)}\n")

        for idx, user in enumerate(users, 1):
            user_id = str(user.id)
            print(f"[{idx}/{len(users)}] Usuário: {user.name} ({user_id})...")

            if not force:
                record = await get_user_recommendations_record(session, user_id)
                has_data = record and (record.explore_items or record.personas_items)
                if has_data and not record.is_stale:
                    print(f"  ⏭️ Já possui recomendações atualizadas. Pulando.")
                    continue

            print(f"  ⚙️ Calculando recomendações completas (Explore, Upcoming, Personas)...")
            data = await compute_and_save_user_recommendations(user_id, force=True)
            if data:
                print(f"  ✅ Salvo no banco com sucesso! (Explore: {len(data.get('explore_items', []))}, Personas: {len(data.get('personas_items', []))})")
            else:
                print(f"  ⚠️ Não foi possível gerar para este usuário (pode não ter perfil/interações ainda).")

    print("\n" + "=" * 60)
    print("✨ PRÉ-AQUECIMENTO CONCLUÍDO COM SUCESSO!")
    print("=" * 60)

if __name__ == "__main__":
    force_flag = "--force" in sys.argv
    asyncio.run(warmup_all_users(force=force_flag))
