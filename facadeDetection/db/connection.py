from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config.settings import Config

DATABASE_URL = f"sqlite:///{Config.BASE_DIR}/data.db"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
