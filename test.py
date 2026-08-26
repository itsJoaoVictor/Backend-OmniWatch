import asyncio, asyncpg, sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def test():
    try:
        conn = await asyncpg.connect('postgresql://postgres:postgres@127.0.0.1:5432/postgres')
        print("Connected to postgres database via 127.0.0.1")
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(test())
