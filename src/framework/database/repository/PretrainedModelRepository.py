from sqlmodel import Session, select

from framework.database.model.PretrainedModel import PretrainedModel
from framework.database.sql.engine import engine


class PretrainedModelRepository:
    def create(self, model: PretrainedModel):
        with Session(engine) as session:
            session.add(model)
            session.commit()
            return model.id

    def get_by_id(self, id):
        with Session(engine) as session:
            statement = select(PretrainedModel).where(PretrainedModel.id == id)
            model = session.exec(statement).one()
            return model

    def get_by_model_id(self, model_id):
        with Session(engine) as session:
            statement = select(PretrainedModel).where(PretrainedModel.model_id == model_id)
            model = session.exec(statement).one()
            return model


pretrained_model_repository = PretrainedModelRepository()
