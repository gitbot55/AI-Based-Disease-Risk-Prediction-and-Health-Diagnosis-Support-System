import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Tuple

import sqlalchemy
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import event, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)

# ─── SQLAlchemy Instance ──────────────────────────────────────────────────────
db = SQLAlchemy()


# ─── SQLite Pragmas ───────────────────────────────────────────────────────────

@event.listens_for(Engine, "connect")
def set_sqlite_pragmas(dbapi_connection, connection_record):
    """
    Applies SQLite-specific pragmas on every connection.
    Safely ignored if a non-SQLite driver is used in the future.
    """
    if "sqlite" not in type(dbapi_connection).__module__.lower():
        return  # Only apply for SQLite databases
        
    cursor = dbapi_connection.cursor()
    pragmas = [
        "PRAGMA journal_mode = WAL",
        "PRAGMA synchronous  = NORMAL",
        "PRAGMA foreign_keys = ON",
        "PRAGMA busy_timeout = 5000",
        "PRAGMA cache_size   = -32768",
    ]
    try:
        for pragma in pragmas:
            cursor.execute(pragma)
    except Exception as exc:
        logger.warning("Failed to set SQLite pragmas: %s", exc)
    finally:
        cursor.close()


# ─── Initialisation ───────────────────────────────────────────────────────────

def init_db(app: Flask) -> None:
    """Initialize the database layer with the Flask application context."""
    start_time = time.perf_counter()
    db.init_app(app)

    with app.app_context():
        _configure_engine(app)
        _import_models()
        _create_schema()
        _upgrade_schema(app)
        _register_jwt_blocklist(app)
        _verify_connection()

    duration_ms = (time.perf_counter() - start_time) * 1000
    safe_uri = _mask_db_uri(app.config.get("SQLALCHEMY_DATABASE_URI", ""))
    logger.info("Database initialised | uri=%s | duration_ms=%.2f", safe_uri, duration_ms)


def shutdown_db(app: Optional[Flask] = None) -> None:
    """
    Gracefully close database connections and return them to the pool.
    Suitable for signal handlers and application shutdown routines.
    """
    try:
        db.session.remove()
        if db.engine:
            db.engine.dispose()
            logger.info("Database engine disposed and connections safely closed.")
    except Exception as exc:
        logger.exception("Error during graceful database shutdown: %s", exc)


def _mask_db_uri(uri: str) -> str:
    """Mask credentials from database URI for secure logging."""
    if not uri:
        return "None"
    try:
        parsed = make_url(uri)
        if parsed.password:
            return uri.replace(parsed.password, "********")
    except Exception:
        pass
    return uri


def _configure_engine(app: Flask) -> None:
    """
    Configure the SQLAlchemy engine with configurable connection pooling.
    Prepared for scaling to Postgres/MySQL, while keeping SQLite stable.
    """
    if app.config.get("TESTING"):
        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
        app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
            "connect_args": {"check_same_thread": False},
            "poolclass":    StaticPool,
        }
    else:
        db_dir = app.config.get("DATABASE_DIR",
                                 os.path.join(os.path.dirname(os.path.abspath(__file__))))
        if "sqlite:///" in app.config.get("SQLALCHEMY_DATABASE_URI", ""):
            os.makedirs(db_dir, exist_ok=True)
            
        # Defaults optimized for production readiness
        app.config.setdefault("SQLALCHEMY_ENGINE_OPTIONS", {
            "connect_args":         {"check_same_thread": False} if "sqlite" in app.config.get("SQLALCHEMY_DATABASE_URI", "") else {},
            "pool_pre_ping":        app.config.get("DB_POOL_PRE_PING", True),
            "pool_recycle":         app.config.get("DB_POOL_RECYCLE", 1800),
            "pool_timeout":         app.config.get("DB_POOL_TIMEOUT", 30),
            "pool_reset_on_return": app.config.get("DB_POOL_RESET", "rollback"),
        })


def _import_models() -> None:
    from .models import User, Prediction, TokenBlocklist  # noqa: F401


def _create_schema() -> None:
    try:
        db.create_all()
        logger.info("Schema created / verified successfully.")
    except SQLAlchemyError as exc:
        logger.exception("Schema creation failed: %s", exc)
        raise

def _upgrade_schema(app: Flask) -> None:
    """
    Compare ORM model columns with actual database columns.
    If they differ, list missing columns.
    Apply migration (SQLite dev only).
    Refuse startup only if the mismatch cannot be resolved safely.
    """
    try:
        inspector = sqlalchemy.inspect(db.engine)
        tables = inspector.get_table_names()
        
        # We only care about models defined in our application
        from .models import User, Prediction, TokenBlocklist
        models = [User, Prediction, TokenBlocklist]
        
        for model_class in models:
            table_name = model_class.__tablename__
            if table_name not in tables:
                continue
                
            db_columns = {col["name"] for col in inspector.get_columns(table_name)}
            orm_columns = {mapper.key for mapper in model_class.__mapper__.columns}
            
            missing_in_db = orm_columns - db_columns
            
            if missing_in_db:
                is_sqlite = "sqlite" in app.config.get("SQLALCHEMY_DATABASE_URI", "").lower()
                is_dev = app.config.get("ENV") == "development"
                
                if is_sqlite and is_dev:
                    with db.engine.connect() as conn:
                        for col_name in missing_in_db:
                            col_obj = model_class.__mapper__.columns[col_name]
                            col_type = col_obj.type.compile(db.engine.dialect)
                            conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}"))
                        conn.commit()
                    logger.info("Schema upgrade completed for table '%s'. Added columns: %s", table_name, ", ".join(missing_in_db))
                else:
                    logger.error("Schema mismatch in '%s'. Missing columns: %s. Cannot resolve automatically in non-development/non-SQLite environment.", table_name, missing_in_db)
                    raise RuntimeError(f"Database schema mismatch in {table_name}")
    except Exception as exc:
        logger.error("Schema validation/migration failed: %s", exc)
        raise


def _register_jwt_blocklist(app: Flask) -> None:
    from .models import TokenBlocklist

    jwt = app.extensions.get("flask-jwt-extended")
    if not jwt:
        logger.warning("JWTManager not found — blocklist loader skipped.")
        return

    @jwt.token_in_blocklist_loader
    def check_if_token_revoked(jwt_header: dict, jwt_payload: dict) -> bool:
        jti = jwt_payload.get("jti")
        if not jti:
            return False
        try:
            return TokenBlocklist.is_revoked(jti)
        except SQLAlchemyError as exc:
            logger.error("Database error during token blocklist check (fail secure): %s", exc)
            return True  # fail secure

    logger.debug("JWT blocklist loader registered.")


def _verify_connection() -> None:
    start_time = time.perf_counter()
    try:
        db.session.execute(text("SELECT 1"))
        logger.debug("Database connection verified in %.2fms", (time.perf_counter() - start_time) * 1000)
    except OperationalError as exc:
        logger.exception("Database connection failed during verification: %s", exc)
        raise


# ─── Retry Mechanisms ─────────────────────────────────────────────────────────

def execute_with_retry(func: Callable, max_retries: int = 3, base_delay_ms: int = 200) -> Any:
    """
    Executes a database operation with exponential backoff for transient OperationalErrors (e.g., SQLite locking).
    IntegrityErrors are NEVER retried.
    """
    attempt = 1
    while attempt <= max_retries:
        try:
            return func()
        except OperationalError as exc:
            if attempt == max_retries:
                logger.error("Operation failed after %d retries: %s", max_retries, exc)
                raise
            delay = (base_delay_ms * (2 ** (attempt - 1))) / 1000.0
            logger.warning("Transient DB lock/error. Retrying %d/%d in %.2fs. Error: %s", attempt, max_retries, delay, exc)
            time.sleep(delay)
            attempt += 1
        except Exception:
            # Do not retry IntegrityErrors or programming errors
            raise


# ─── Session Utilities ────────────────────────────────────────────────────────

@contextmanager
def managed_session(nested: bool = False):
    """
    Context manager for transactional control outside request context.
    Auto-commits on success, rolls back on any exception.
    
    Args:
        nested: If True, uses a SAVEPOINT via session.begin_nested() 
                to allow sub-transaction rollbacks without failing the parent.
    """
    session = db.session
    if nested:
        session.begin_nested()
        
    start_time = time.perf_counter()
    logger.debug("Transaction started%s", " (nested)" if nested else "")
    
    try:
        yield session
        session.commit()
        logger.debug("Transaction committed successfully in %.2fms", (time.perf_counter() - start_time) * 1000)
    except IntegrityError as exc:
        session.rollback()
        logger.warning("IntegrityError — transaction rolled back: %s", exc.orig)
        raise
    except SQLAlchemyError as exc:
        session.rollback()
        logger.exception("SQLAlchemyError — transaction rolled back: %s", exc)
        raise
    finally:
        if not nested:
            session.close()


def safe_commit(max_retries: int = 3) -> bool:
    """
    Commit the current session with automatic rollback and retries for transient locks.
    Returns True on success, False on failure.
    """
    def _do_commit():
        start_time = time.perf_counter()
        db.session.commit()
        logger.debug("Session committed safely in %.2fms", (time.perf_counter() - start_time) * 1000)
        return True

    try:
        return execute_with_retry(_do_commit, max_retries=max_retries)
    except IntegrityError as exc:
        db.session.rollback()
        logger.warning("safe_commit IntegrityError (rollback triggered): %s", exc.orig)
        return False
    except SQLAlchemyError as exc:
        db.session.rollback()
        logger.exception("safe_commit SQLAlchemyError (rollback triggered): %s", exc)
        return False


def paginate_query(query, page: int, per_page: int, max_per_page: int = 100) -> dict:
    """Apply pagination and return items + metadata dict."""
    per_page  = max(1, min(int(per_page), max_per_page))
    page      = max(1, int(page))
    
    start_time = time.perf_counter()
    paginated = query.paginate(page=page, per_page=per_page, error_out=False)
    
    logger.debug("Query paginated (page=%d, per_page=%d) in %.2fms", page, per_page, (time.perf_counter() - start_time) * 1000)

    return {
        "items":     paginated.items,
        "total":     paginated.total,
        "page":      paginated.page,
        "per_page":  paginated.per_page,
        "pages":     paginated.pages,
        "has_next":  paginated.has_next,
        "has_prev":  paginated.has_prev,
        "next_page": paginated.next_num if paginated.has_next else None,
        "prev_page": paginated.prev_num if paginated.has_prev else None,
    }


# ─── Health Check ─────────────────────────────────────────────────────────────

def get_db_health() -> dict:
    """Return comprehensive database health metrics and operational status."""
    from datetime import datetime, timezone
    start_time = time.perf_counter()

    result = {
        "status":        "unhealthy",
        "tables":        [],
        "checked_at":    datetime.now(timezone.utc).isoformat(),
        "error":         None,
        "latency_ms":    0.0,
        "database_type": "unknown",
        "engine_info":   {
            "sqlalchemy_version": sqlalchemy.__version__,
            "pool_status": "unavailable"
        }
    }
    
    try:
        db.session.execute(text("SELECT 1"))
        latency = (time.perf_counter() - start_time) * 1000
        
        engine = db.engine
        result["database_type"] = engine.name
        
        # Pool stats (not all pool types support this)
        try:
            result["engine_info"]["pool_status"] = engine.pool.status()
        except Exception:
            pass
            
        tables = db.inspect(engine).get_table_names()
        
        result["status"] = "healthy"
        result["tables"] = sorted(tables)
        result["latency_ms"] = round(latency, 2)
        
        logger.debug("Health check verified %s in %.2fms", engine.name, latency)
    except OperationalError as exc:
        result["error"] = "Database connection failed or locked"
        logger.error("DB health check OperationalError: %s", exc)
    except SQLAlchemyError as exc:
        result["error"] = str(exc)
        logger.error("DB health check failed: %s", exc)

    return result