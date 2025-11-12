from fastapi import FastAPI, Request, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import time
import json
import pandas as pd
import joblib

# OpenTelemetry
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# tracer setup
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# logging setup
logger = logging.getLogger("iris-log-ml-service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(json.dumps({
    "severity": "%(levelname)s",
    "message": "%(message)s",
    "timestamp": "%(asctime)s"
}))
handler.setFormatter(formatter)
logger.addHandler(handler)

app = FastAPI(title="🌸 Iris Classifier API with Logging & Tracing")

app_state = {"is_ready": False, "is_alive": True}
model = None


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = joblib.load("model.joblib")
        app_state["is_ready"] = True
        logger.info(json.dumps({"event": "model_loaded", "status": "success"}))
    except Exception as e:
        app_state["is_ready"] = False
        logger.exception(json.dumps({"event": "model_load_failed", "error": str(e)}))


@app.get("/live_check")
async def live_check():
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.get("/ready_check")
async def ready_check():
    if app_state["is_ready"]:
        return {"status": "ready"}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


@app.middleware("http")
async def add_timing(request: Request, call_next):
    start = time.time()
    resp = await call_next(request)
    resp.headers["X-Process-Time-ms"] = str(round((time.time() - start) * 1000, 2))
    return resp


@app.exception_handler(Exception)
async def global_exception(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")
    logger.exception(json.dumps({
        "event": "unhandled_exception",
        "trace_id": trace_id,
        "path": str(request.url),
        "error": str(exc)
    }))
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error", "trace_id": trace_id})


@app.get("/")
def root():
    return {"message": "Welcome to Iris API"}


@app.post("/predict/")
async def predict(data: IrisInput, request: Request):
    if not app_state["is_ready"]:
        raise HTTPException(status_code=503, detail="Model not ready")

    with tracer.start_as_current_span("iris_inference") as span:
        df = pd.DataFrame([data.dict()])
        pred = model.predict(df)[0]
        trace_id = format(span.get_span_context().trace_id, "032x")
        logger.info({
            "event": "prediction",
            "input": data.dict(),
            "prediction": str(pred),
            "trace_id": trace_id
        })
        return {"predicted_class": str(pred), "trace_id": trace_id}
