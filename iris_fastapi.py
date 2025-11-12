from fastapi import FastAPI, Request, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import time
import json
import os
import sys

import pandas as pd
import joblib

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# -----------------------------------------------------------------------------
# Tracer setup
# -----------------------------------------------------------------------------
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# -----------------------------------------------------------------------------
# Logging setup – IMPORTANT: no outer JSON wrapper
# -----------------------------------------------------------------------------
logger = logging.getLogger("iris-log-ml-service")
logger.setLevel(logging.INFO)

# send to stdout so GKE/Cloud Logging treat it as normal INFO
handler = logging.StreamHandler(stream=sys.stdout)
# we will log complete JSON ourselves
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="🌸 Iris Classifier API with Logging & Tracing")

app_state = {"is_ready": False, "is_alive": True}
model = None

# use the real project id at runtime (GKE sets this), fallback for local
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "precise-braid-474114-m3")


# -----------------------------------------------------------------------------
# Schema
# -----------------------------------------------------------------------------
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# -----------------------------------------------------------------------------
# Startup – load model
# -----------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = joblib.load("model.joblib")
        app_state["is_ready"] = True
        logger.info(json.dumps({
            "event": "model_loaded",
            "status": "success"
        }))
    except Exception as e:
        app_state["is_ready"] = False
        logger.error(json.dumps({
            "event": "model_load_failed",
            "error": str(e)
        }))


# -----------------------------------------------------------------------------
# Probes
# -----------------------------------------------------------------------------
@app.get("/live_check", tags=["Probe"])
async def liveness_probe():
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/ready_check", tags=["Probe"])
async def readiness_probe():
    if app_state["is_ready"]:
        return {"status": "ready"}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


# -----------------------------------------------------------------------------
# Middleware
# -----------------------------------------------------------------------------
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Process-Time-ms"] = str(duration)
    return response


# -----------------------------------------------------------------------------
# Global exception handler
# -----------------------------------------------------------------------------
@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    span_ctx = span.get_span_context()
    trace_id_hex = format(span_ctx.trace_id, "032x")
    span_id_hex = format(span_ctx.span_id, "016x")
    gcp_trace = f"projects/{PROJECT_ID}/traces/{trace_id_hex}"

    logger.error(json.dumps({
        "event": "unhandled_exception",
        "trace": gcp_trace,
        "spanId": span_id_hex,
        "traceSampled": True,
        "path": str(request.url),
        "error": str(exc)
    }))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "trace_id": trace_id_hex},
    )


# -----------------------------------------------------------------------------
# Root
# -----------------------------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classifier API with logging!"}


# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------
@app.post("/predict")
async def predict_iris(data: IrisInput, request: Request):
    if not app_state["is_ready"]:
        raise HTTPException(status_code=503, detail="Model not ready")

    start_time = time.time()

    # start a span – this creates the trace/span in Cloud Trace
    with tracer.start_as_current_span("model_inference") as span:
        span_ctx = span.get_span_context()
        trace_id_hex = format(span_ctx.trace_id, "032x")
        span_id_hex = format(span_ctx.span_id, "016x")
        gcp_trace = f"projects/{PROJECT_ID}/traces/{trace_id_hex}"

        try:
            # model inference
            df = pd.DataFrame([data.dict()])
            pred = model.predict(df)[0]

            latency = round((time.time() - start_time) * 1000, 2)

            # 👇 this is the log Cloud Logging can link to this trace
            logger.info(json.dumps({
                "event": "prediction",
                "trace": gcp_trace,          # link to Cloud Trace
                "spanId": span_id_hex,       # which span
                "traceSampled": True,
                "input": data.dict(),
                "result": {"predicted_class": str(pred)},
                "latency_ms": latency,
                "status": "success",
                "path": str(request.url),
            }))

            return {"predicted_class": str(pred), "trace_id": trace_id_hex}

        except Exception as e:
            latency = round((time.time() - start_time) * 1000, 2)
            logger.error(json.dumps({
                "event": "prediction_error",
                "trace": gcp_trace,
                "spanId": span_id_hex,
                "traceSampled": True,
                "error": str(e),
                "latency_ms": latency,
                "status": "error",
                "path": str(request.url),
            }))
            raise HTTPException(status_code=500, detail="Prediction failed")
