from pathlib import Path


def test_alembic_migration_files_exist() -> None:
    assert Path("alembic.ini").exists()
    assert Path("app/db/migrations/env.py").exists()
    assert Path("app/db/migrations/versions/0001_initial_schema.py").exists()
    assert Path("app/db/migrations/versions/0002_chunk_structural_fields.py").exists()
    assert Path("app/db/migrations/versions/0003_chunk_token_count.py").exists()
