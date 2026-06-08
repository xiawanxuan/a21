import os
import sys
import json
import asyncio
from typing import List, Optional, Dict, Any
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from src.models import build_model
from src.inference_visualization import predict_single, predict_batch, load_model
from src.spectral_analysis import (
    detect_spectral_lines,
    compute_spectral_features,
    detect_anomalous_spectrum,
    batch_detect_lines,
    batch_detect_anomalies,
)

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


class SpectrumInput(BaseModel):
    spectrum: List[float] = Field(..., description="光谱数据，一维数组")
    wavelength: Optional[List[float]] = Field(None, description="波长数组（可选）")


class BatchSpectrumInput(BaseModel):
    spectra: List[List[float]] = Field(..., description="批量光谱数据")
    wavelength: Optional[List[float]] = Field(None, description="波长数组（可选）")


class LineDetectionParams(BaseModel):
    height_threshold: float = 2.0
    distance: int = 10
    min_prominence: float = 0.5
    window_size: int = 5


class AnomalyDetectionParams(BaseModel):
    threshold: float = 3.0
    reference_stats: Optional[Dict[str, Dict[str, float]]] = None


def create_app(config=None, model=None):
    if config is None:
        config = Config()

    if model is None:
        model = build_model(config)
        model = model.to(config.device)
        model.eval()

    app = FastAPI(
        title="天文光谱红移测量与分类 API",
        description="基于 Transformer 的天文光谱红移自动测量与分类系统 API 服务",
        version="1.0.0",
    )

    app.state.config = config
    app.state.model = model

    @app.get("/health", summary="健康检查")
    async def health_check():
        return {
            "status": "ok",
            "model_loaded": True,
            "device": str(config.device),
        }

    @app.get("/config", summary="获取模型配置")
    async def get_config():
        return {
            "spectrum_length": config.spectrum_length,
            "num_classes": config.num_classes,
            "class_names": config.class_names,
            "d_model": config.d_model,
            "nhead": config.nhead,
            "num_encoder_layers": config.num_encoder_layers,
        }

    @app.post("/predict/single", summary="单条光谱推理")
    async def predict_single_spectrum(input_data: SpectrumInput):
        try:
            spectrum = np.array(input_data.spectrum, dtype=np.float32)

            if len(spectrum) != config.spectrum_length:
                raise HTTPException(
                    status_code=400,
                    detail=f"Spectrum length must be {config.spectrum_length}, got {len(spectrum)}"
                )

            result = predict_single(model, spectrum, config, device=config.device)

            return {
                "success": True,
                "redshift": result["redshift"],
                "classification": {
                    "class_index": result["class_index"],
                    "class_name": result["class_name"],
                    "probabilities": result["class_probabilities"],
                },
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict/batch", summary="批量光谱推理")
    async def predict_batch_spectra(input_data: BatchSpectrumInput):
        try:
            spectra = np.array(input_data.spectra, dtype=np.float32)

            if spectra.shape[1] != config.spectrum_length:
                raise HTTPException(
                    status_code=400,
                    detail=f"Spectrum length must be {config.spectrum_length}, got {spectra.shape[1]}"
                )

            results = predict_batch(model, spectra, config, device=config.device)

            formatted_results = []
            for r in results:
                formatted_results.append({
                    "redshift": r["redshift"],
                    "classification": {
                        "class_index": r["class_index"],
                        "class_name": r["class_name"],
                        "probabilities": r["class_probabilities"],
                    },
                })

            return {
                "success": True,
                "count": len(formatted_results),
                "results": formatted_results,
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/analyze/lines", summary="谱线检测")
    async def analyze_lines(
        input_data: SpectrumInput,
        params: LineDetectionParams = LineDetectionParams(),
    ):
        try:
            spectrum = np.array(input_data.spectrum, dtype=np.float32)
            wavelength = np.array(input_data.wavelength) if input_data.wavelength else None

            result = detect_spectral_lines(
                spectrum,
                wavelength=wavelength,
                height_threshold=params.height_threshold,
                distance=params.distance,
                min_prominence=params.min_prominence,
                window_size=params.window_size,
            )

            return {
                "success": True,
                "num_emission_lines": result["num_emission"],
                "num_absorption_lines": result["num_absorption"],
                "emission_lines": result["emission_lines"][:20],
                "absorption_lines": result["absorption_lines"][:20],
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/analyze/lines/batch", summary="批量谱线检测")
    async def analyze_lines_batch(
        input_data: BatchSpectrumInput,
        params: LineDetectionParams = LineDetectionParams(),
    ):
        try:
            spectra = np.array(input_data.spectra, dtype=np.float32)
            wavelength = np.array(input_data.wavelength) if input_data.wavelength else None

            results = batch_detect_lines(
                spectra,
                wavelength=wavelength,
                height_threshold=params.height_threshold,
                distance=params.distance,
                min_prominence=params.min_prominence,
                window_size=params.window_size,
            )

            summary = []
            for r in results:
                summary.append({
                    "num_emission_lines": r["num_emission"],
                    "num_absorption_lines": r["num_absorption"],
                    "top_emission": r["emission_lines"][:5],
                    "top_absorption": r["absorption_lines"][:5],
                })

            return {
                "success": True,
                "count": len(summary),
                "results": summary,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/analyze/anomaly", summary="异常光谱检测")
    async def analyze_anomaly(
        input_data: SpectrumInput,
        params: AnomalyDetectionParams = AnomalyDetectionParams(),
    ):
        try:
            spectrum = np.array(input_data.spectrum, dtype=np.float32)
            features, line_result = compute_spectral_features(spectrum)

            result = detect_anomalous_spectrum(
                spectrum,
                features=features,
                threshold=params.threshold,
                reference_stats=params.reference_stats,
            )

            return {
                "success": True,
                "is_anomalous": result["is_anomalous"],
                "anomaly_score": result["anomaly_score"],
                "reasons": result["reasons"],
                "features": features,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/analyze/anomaly/batch", summary="批量异常光谱检测")
    async def analyze_anomaly_batch(
        input_data: BatchSpectrumInput,
        params: AnomalyDetectionParams = AnomalyDetectionParams(),
    ):
        try:
            spectra = np.array(input_data.spectra, dtype=np.float32)

            results, ref_stats = batch_detect_anomalies(
                spectra,
                reference_stats=params.reference_stats,
                threshold=params.threshold,
            )

            anomaly_count = sum(1 for r in results if r["is_anomalous"])

            return {
                "success": True,
                "count": len(results),
                "anomaly_count": anomaly_count,
                "anomaly_rate": float(anomaly_count / len(results)),
                "reference_stats": ref_stats,
                "results": results,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/analyze/full", summary="综合分析：红移+分类+谱线+异常")
    async def full_analysis(input_data: SpectrumInput):
        try:
            spectrum = np.array(input_data.spectrum, dtype=np.float32)

            if len(spectrum) != config.spectrum_length:
                raise HTTPException(
                    status_code=400,
                    detail=f"Spectrum length must be {config.spectrum_length}, got {len(spectrum)}"
                )

            wavelength = np.array(input_data.wavelength) if input_data.wavelength else None

            pred_result = predict_single(model, spectrum, config, device=config.device)
            line_result = detect_spectral_lines(spectrum, wavelength=wavelength)
            features, _ = compute_spectral_features(spectrum)
            anomaly_result = detect_anomalous_spectrum(spectrum, features=features)

            return {
                "success": True,
                "redshift": pred_result["redshift"],
                "classification": {
                    "class_index": pred_result["class_index"],
                    "class_name": pred_result["class_name"],
                    "probabilities": pred_result["class_probabilities"],
                },
                "spectral_lines": {
                    "num_emission": line_result["num_emission"],
                    "num_absorption": line_result["num_absorption"],
                    "emission_lines": line_result["emission_lines"][:10],
                    "absorption_lines": line_result["absorption_lines"][:10],
                },
                "anomaly_detection": {
                    "is_anomalous": anomaly_result["is_anomalous"],
                    "anomaly_score": anomaly_result["anomaly_score"],
                    "reasons": anomaly_result["reasons"],
                },
                "features": features,
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/analyze/full/batch", summary="批量综合分析")
    async def full_analysis_batch(input_data: BatchSpectrumInput):
        try:
            spectra = np.array(input_data.spectra, dtype=np.float32)

            if spectra.shape[1] != config.spectrum_length:
                raise HTTPException(
                    status_code=400,
                    detail=f"Spectrum length must be {config.spectrum_length}, got {spectra.shape[1]}"
                )

            wavelength = np.array(input_data.wavelength) if input_data.wavelength else None

            pred_results = predict_batch(model, spectra, config, device=config.device)
            line_results = batch_detect_lines(spectra, wavelength=wavelength)
            anomaly_results, ref_stats = batch_detect_anomalies(spectra)

            combined = []
            for i in range(len(spectra)):
                combined.append({
                    "redshift": pred_results[i]["redshift"],
                    "classification": {
                        "class_index": pred_results[i]["class_index"],
                        "class_name": pred_results[i]["class_name"],
                        "probabilities": pred_results[i]["class_probabilities"],
                    },
                    "spectral_lines": {
                        "num_emission": line_results[i]["num_emission"],
                        "num_absorption": line_results[i]["num_absorption"],
                    },
                    "anomaly_detection": {
                        "is_anomalous": anomaly_results[i]["is_anomalous"],
                        "anomaly_score": anomaly_results[i]["anomaly_score"],
                        "reasons": anomaly_results[i]["reasons"],
                    },
                })

            anomaly_count = sum(1 for r in anomaly_results if r["is_anomalous"])

            return {
                "success": True,
                "count": len(combined),
                "anomaly_count": anomaly_count,
                "anomaly_rate": float(anomaly_count / len(combined)),
                "results": combined,
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


def start_server(config=None, checkpoint_path=None, host="0.0.0.0", port=8000):
    if not FASTAPI_AVAILABLE:
        print("Error: FastAPI is not installed. Please install it with: pip install fastapi uvicorn")
        return

    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is not installed. Please install it with: pip install uvicorn")
        return

    if config is None:
        config = Config()

    if checkpoint_path:
        model = load_model(config, checkpoint_path)
    else:
        model = build_model(config)
        model = model.to(config.device)
        model.eval()

    app = create_app(config, model)

    print(f"Starting API server on {host}:{port}")
    print(f"  Docs: http://{host}:{port}/docs")
    print(f"  Redoc: http://{host}:{port}/redoc")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
