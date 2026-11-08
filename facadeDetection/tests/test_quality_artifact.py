from contextlib import contextmanager

import numpy as np
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from models import Base, Facade, Project
from services.dal.results_repo import ResultsRepo


@contextmanager
def memory_session(Session, _uuid):
    session = Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def test_artifact_path_keeps_large_indices_out_of_quality_json(monkeypatch, tmp_path):
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(
        'services.dal.results_repo.project_session',
        lambda uuid: memory_session(Session, uuid))
    with memory_session(Session, 'uuid') as session:
        session.add(Project(uuid='uuid', name='artifact'))

    quality = {
        'ok': True,
        'overall': {},
        '__global_indices': np.arange(10_000, dtype=np.int64),
    }
    artifact = ResultsRepo.persist_quality_artifact(tmp_path, 1, quality)
    quality.pop('__global_indices')
    ResultsRepo.commit_quality_success(
        'uuid', 1, quality, display_no=1,
        facade_data={'proxy_indices': [0, 1, 2]},
        quality_artifact_path=artifact)

    with memory_session(Session, 'uuid') as session:
        report = session.execute(select(Facade)).scalar_one().quality_report_json
    assert '__global_indices' not in report
    assert report['quality_artifact_path'] == artifact
    restored = ResultsRepo.load_quality_artifact(tmp_path / artifact)
    assert np.array_equal(restored, np.arange(10_000))
