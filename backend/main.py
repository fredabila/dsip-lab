from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.detection import router
from services.inference import ppe_service


@asynccontextmanager
async def lifespan(app: FastAPI):
	ppe_service.load()
	yield


app = FastAPI(title="Helmax PPE Detection API", lifespan=lifespan)

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_methods=["*"],
	allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def health():
	return {"status": "ok"}
