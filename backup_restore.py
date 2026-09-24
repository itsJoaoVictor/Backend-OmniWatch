"""
Script de Backup, Restauração e Migração do PostgreSQL (OmniWatch)
Suporta credenciais e senhas diferentes entre Origem e Destino,
além de URLs SQLAlchemy com prefixo '+asyncpg'.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit

# Diretório padrão para backups
BACKUP_DIR = Path(__file__).resolve().parent / "backups"

# Configurações padrão (podem ser sobrescritas por variáveis de ambiente ou argumentos de linha de comando)
DEFAULT_SOURCE_URL = os.getenv(
    "SOURCE_DATABASE_URL",
    "postgresql+asyncpg://omniwatch:Jvgf1211@127.0.0.1:5433/omniwatch",
)
DEFAULT_SOURCE_PASSWORD = os.getenv("SOURCE_DB_PASSWORD", "Jvgf1211")

DEFAULT_TARGET_URL = os.getenv(
    "TARGET_DATABASE_URL",
    "postgresql+asyncpg://postgres:jvgf1211@localhost:5432/omniwatch",
)
DEFAULT_TARGET_PASSWORD = os.getenv("TARGET_DB_PASSWORD", "jvgf1211")


def find_pg_tool(tool_name: str) -> str:
    """Localiza o binário do PostgreSQL (ex: pg_dump ou psql) no PATH ou em Program Files."""
    if shutil.which(tool_name):
        return tool_name

    # Procura nas versões instaladas em Program Files (ordem decrescente de versão)
    candidates = sorted(
        glob.glob(rf"C:\Program Files\PostgreSQL\*\bin\{tool_name}.exe"),
        reverse=True,
    )
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Binário '{tool_name}' não encontrado no PATH nem em 'C:\\Program Files\\PostgreSQL\\*\\bin'. "
        "Verifique se o PostgreSQL está instalado."
    )


def parse_db_url(url: str, override_password: str = None) -> dict:
    """Extrai usuário, senha, host, porta e banco de uma URL de conexão."""
    # Remove prefixos como '+asyncpg' ou '+psycopg'
    clean_url = url
    if "://" in clean_url:
        scheme, rest = clean_url.split("://", 1)
        base_scheme = scheme.split("+")[0]
        clean_url = f"{base_scheme}://{rest}"

    parsed = urlsplit(clean_url)
    username = unquote(parsed.username) if parsed.username else "postgres"
    password = override_password if override_password else (unquote(parsed.password) if parsed.password else "")
    host = parsed.hostname or "127.0.0.1"
    port = str(parsed.port or 5432)
    dbname = parsed.path.lstrip("/") or "omniwatch"

    return {
        "user": username,
        "password": password,
        "host": host,
        "port": port,
        "dbname": dbname,
    }


def test_connection(db_info: dict, label: str) -> bool:
    """Testa se a conexão com o banco de dados é bem-sucedida."""
    psql_path = find_pg_tool("psql")
    env = os.environ.copy()
    if db_info["password"]:
        env["PGPASSWORD"] = db_info["password"]

    cmd = [
        psql_path,
        "-h", db_info["host"],
        "-p", db_info["port"],
        "-U", db_info["user"],
        "-d", db_info["dbname"],
        "-c", "SELECT current_user, current_database();",
    ]

    print(f"[*] Testando conexão [{label}] ({db_info['user']}@{db_info['host']}:{db_info['port']}/{db_info['dbname']})...")
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"[+] Conexão [{label}] OK!")
        return True
    else:
        print(f"[-] Falha na conexão [{label}]: {res.stderr.strip()}")
        return False


def run_backup(source_info: dict, output_file: Path) -> Path:
    """Gera um dump (.sql) a partir do banco de origem."""
    pg_dump_path = find_pg_tool("pg_dump")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if source_info["password"]:
        env["PGPASSWORD"] = source_info["password"]

    cmd = [
        pg_dump_path,
        "-h", source_info["host"],
        "-p", source_info["port"],
        "-U", source_info["user"],
        "-d", source_info["dbname"],
        "--clean",            # Adiciona comandos DROP TABLE antes de CREATE TABLE
        "--if-exists",        # Evita erros se as tabelas ainda não existirem
        "--no-owner",         # Não força as tabelas a pertencerem ao usuário de origem
        "--no-privileges",    # Evita erros de permissões de roles inexistentes
        "-f", str(output_file),
    ]

    print(f"[*] Iniciando backup da base de origem ({source_info['dbname']})...")
    print(f"    Usuário de origem: {source_info['user']}")
    print(f"    Arquivo de saída:  {output_file}")

    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[-] Erro ao executar pg_dump:\n{res.stderr}")
        sys.exit(res.returncode)

    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"[+] Backup concluído com sucesso! Tamanho: {size_mb:.2f} MB")
    return output_file


def run_restore(target_info: dict, input_file: Path) -> None:
    """Restaura um arquivo .sql no banco de destino."""
    if not input_file.exists():
        print(f"[-] Arquivo de backup não encontrado: {input_file}")
        sys.exit(1)

    psql_path = find_pg_tool("psql")
    env = os.environ.copy()
    if target_info["password"]:
        env["PGPASSWORD"] = target_info["password"]

    cmd = [
        psql_path,
        "-h", target_info["host"],
        "-p", target_info["port"],
        "-U", target_info["user"],
        "-d", target_info["dbname"],
        "-f", str(input_file),
    ]

    print(f"[*] Restaurando no banco de destino ({target_info['dbname']})...")
    print(f"    Usuário de destino: {target_info['user']}")
    print(f"    Arquivo de origem:  {input_file}")

    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[-] Erro ao executar psql:\n{res.stderr}")
        sys.exit(res.returncode)

    print("[+] Restauração concluída com sucesso no banco de destino!")


def interactive_menu():
    print("\n" + "=" * 55)
    print("    OmniWatch - Utilitário de Backup & Migração")
    print("=" * 55)
    print("1) Fazer Backup da Origem (Gera arquivo .sql)")
    print("2) Sincronizar Direto (Origem -> Destino)")
    print("3) Restaurar arquivo .sql no Destino")
    print("4) Testar Conexões (Origem e Destino)")
    print("0) Sair")
    print("=" * 55)
    choice = input("Escolha uma opção [0-4]: ").strip()

    if choice == "1":
        source_url = input(f"URL de Origem [{DEFAULT_SOURCE_URL}]: ").strip() or DEFAULT_SOURCE_URL
        source_pwd = input(f"Senha de Origem [{DEFAULT_SOURCE_PASSWORD}]: ").strip() or DEFAULT_SOURCE_PASSWORD
        source_info = parse_db_url(source_url, source_pwd)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = BACKUP_DIR / f"{source_info['dbname']}_backup_{timestamp}.sql"
        run_backup(source_info, output_file)

    elif choice == "2":
        source_url = input(f"URL de Origem [{DEFAULT_SOURCE_URL}]: ").strip() or DEFAULT_SOURCE_URL
        source_pwd = input(f"Senha de Origem [{DEFAULT_SOURCE_PASSWORD}]: ").strip() or DEFAULT_SOURCE_PASSWORD
        target_url = input(f"URL de Destino [{DEFAULT_TARGET_URL}]: ").strip() or DEFAULT_TARGET_URL
        target_pwd = input(f"Senha de Destino [{DEFAULT_TARGET_PASSWORD}]: ").strip() or DEFAULT_TARGET_PASSWORD
        source_info = parse_db_url(source_url, source_pwd)
        target_info = parse_db_url(target_url, target_pwd)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_file = BACKUP_DIR / f"{source_info['dbname']}_sync_{timestamp}.sql"
        print("\n=== ETAPA 1: DUMP DA ORIGEM ===")
        run_backup(source_info, temp_file)
        print("\n=== ETAPA 2: RESTORE NO DESTINO ===")
        run_restore(target_info, temp_file)

    elif choice == "3":
        sql_files = sorted(BACKUP_DIR.glob("*.sql"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not sql_files:
            print(f"[-] Nenhum arquivo .sql encontrado em {BACKUP_DIR}")
            return
        print("\nArquivos de backup disponíveis:")
        for idx, f in enumerate(sql_files, 1):
            print(f"  {idx}) {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")
        file_choice = input(f"Escolha o arquivo [1-{len(sql_files)}]: ").strip()
        try:
            chosen_file = sql_files[int(file_choice) - 1]
        except (ValueError, IndexError):
            print("[-] Opção inválida.")
            return

        target_url = input(f"URL de Destino [{DEFAULT_TARGET_URL}]: ").strip() or DEFAULT_TARGET_URL
        target_pwd = input(f"Senha de Destino [{DEFAULT_TARGET_PASSWORD}]: ").strip() or DEFAULT_TARGET_PASSWORD
        target_info = parse_db_url(target_url, target_pwd)
        run_restore(target_info, chosen_file)

    elif choice == "4":
        source_url = input(f"URL de Origem [{DEFAULT_SOURCE_URL}]: ").strip() or DEFAULT_SOURCE_URL
        source_pwd = input(f"Senha de Origem [{DEFAULT_SOURCE_PASSWORD}]: ").strip() or DEFAULT_SOURCE_PASSWORD
        target_url = input(f"URL de Destino [{DEFAULT_TARGET_URL}]: ").strip() or DEFAULT_TARGET_URL
        target_pwd = input(f"Senha de Destino [{DEFAULT_TARGET_PASSWORD}]: ").strip() or DEFAULT_TARGET_PASSWORD
        source_info = parse_db_url(source_url, source_pwd)
        target_info = parse_db_url(target_url, target_pwd)
        s_ok = test_connection(source_info, "ORIGEM")
        t_ok = test_connection(target_info, "DESTINO")
        if s_ok and t_ok:
            print("\n[OK] Ambas as conexões foram bem-sucedidas!")
        else:
            print("\n[AVISO] Verifique as credenciais que falharam acima.")

    elif choice == "0":
        print("Saindo...")
    else:
        print("Opção inválida.")


def main():
    if len(sys.argv) == 1:
        interactive_menu()
        return

    parser = argparse.ArgumentParser(
        description="Utilitário de backup e migração de banco de dados PostgreSQL."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcomando: backup
    backup_parser = subparsers.add_parser("backup", help="Gera um backup (.sql) da base de origem")
    backup_parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL, help="URL do banco de origem")
    backup_parser.add_argument("--source-password", default=DEFAULT_SOURCE_PASSWORD, help="Senha do banco de origem")
    backup_parser.add_argument("--out", default=None, help="Caminho do arquivo .sql de saída")

    # Subcomando: restore
    restore_parser = subparsers.add_parser("restore", help="Restaura um arquivo (.sql) na base de destino")
    restore_parser.add_argument("file", help="Caminho do arquivo .sql para restaurar")
    restore_parser.add_argument("--target-url", default=DEFAULT_TARGET_URL, help="URL do banco de destino")
    restore_parser.add_argument("--target-password", default=DEFAULT_TARGET_PASSWORD, help="Senha do banco de destino")

    # Subcomando: sync (dump da origem + restore no destino)
    sync_parser = subparsers.add_parser("sync", help="Faz o dump da origem e restaura diretamente no destino")
    sync_parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL, help="URL do banco de origem")
    sync_parser.add_argument("--source-password", default=DEFAULT_SOURCE_PASSWORD, help="Senha do banco de origem")
    sync_parser.add_argument("--target-url", default=DEFAULT_TARGET_URL, help="URL do banco de destino")
    sync_parser.add_argument("--target-password", default=DEFAULT_TARGET_PASSWORD, help="Senha do banco de destino")

    # Subcomando: test (testa as conexões)
    test_parser = subparsers.add_parser("test", help="Testa conectividade com a origem e com o destino")
    test_parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL, help="URL do banco de origem")
    test_parser.add_argument("--source-password", default=DEFAULT_SOURCE_PASSWORD, help="Senha do banco de origem")
    test_parser.add_argument("--target-url", default=DEFAULT_TARGET_URL, help="URL do banco de destino")
    test_parser.add_argument("--target-password", default=DEFAULT_TARGET_PASSWORD, help="Senha do banco de destino")

    args = parser.parse_args()

    if args.command == "backup":
        source_info = parse_db_url(args.source_url, args.source_password)
        if args.out:
            output_file = Path(args.out)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = BACKUP_DIR / f"{source_info['dbname']}_backup_{timestamp}.sql"
        run_backup(source_info, output_file)

    elif args.command == "restore":
        target_info = parse_db_url(args.target_url, args.target_password)
        input_file = Path(args.file)
        run_restore(target_info, input_file)

    elif args.command == "sync":
        source_info = parse_db_url(args.source_url, args.source_password)
        target_info = parse_db_url(args.target_url, args.target_password)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_file = BACKUP_DIR / f"{source_info['dbname']}_sync_{timestamp}.sql"

        print("=== ETAPA 1: DUMP DA ORIGEM ===")
        run_backup(source_info, temp_file)

        print("\n=== ETAPA 2: RESTORE NO DESTINO ===")
        run_restore(target_info, temp_file)

    elif args.command == "test":
        source_info = parse_db_url(args.source_url, args.source_password)
        target_info = parse_db_url(args.target_url, args.target_password)
        s_ok = test_connection(source_info, "ORIGEM")
        t_ok = test_connection(target_info, "DESTINO")
        if s_ok and t_ok:
            print("\n[OK] Ambas as conexões foram bem-sucedidas!")
        else:
            print("\n[AVISO] Verifique as credenciais que falharam acima.")


if __name__ == "__main__":
    main()
