from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.estimate import router as estimate_router
from src.config.logging import configure_logging
from src.domain.estimation.service import EstimationService, build_estimation_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    service: EstimationService = build_estimation_service()
    app.state.estimation_service = service
    yield


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(
        lifespan=lifespan,
        title="AI-ASSIST Estimation API",
        description=(
            "Real-time power-grid security assessment: given a live grid operating state, "
            "retrieves the most similar historical/simulated states from a vector database "
            "and uses their known transient-stability outcomes to estimate the Critical "
            "Clearing Time (CCT). See each endpoint below for the by-generator vs "
            "by-location report shapes."
        ),
    )
    app.include_router(estimate_router)
    return app


app = create_app()
