"""
Главный FastAPI-бэкенд для демо IAM + Adaptive Auth + UEBA.

Важно:
- Это учебный прототип для live-презентации.
- Хранение данных сделано в памяти (без БД), чтобы код был максимально простым.
- Аутентификация и MFA здесь демонстрационные, не production-ready.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from adaptive_auth import DEFAULT_COUNTRY, calculate_auth_risk, get_country_from_ip


# ------------------------------------------------------------
# Пути к данным проекта
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
STATIC_DIR = PROJECT_ROOT / "static"


# ------------------------------------------------------------
# Инициализация приложения
# ------------------------------------------------------------
app = FastAPI(title="IAM Guardian Demo")

# Для локальной разработки разрешаем CORS полностью.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # Для режима "разрешить все" credential-флаг лучше отключить:
    # комбинация allow_origins=["*"] + credentials=True противоречит CORS-спецификации.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем папку со статическими файлами (HTML/CSS/JS).
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ------------------------------------------------------------
# Глобальные переменные для ML-артефактов
# ------------------------------------------------------------
isolation_forest_model: Any = None
scaler: Any = None
minmax_scaler: Any = None
pca_model: Any = None

ml_metrics: dict[str, Any] = {}
feature_importance: list[dict[str, Any]] = []
test_scores_data: list[dict[str, Any]] = []
roc_curve_data: dict[str, Any] = {}
pca_projection_data: list[dict[str, Any]] = []
pr_curve_data: dict[str, Any] = {}
threshold_analysis_data: list[dict[str, Any]] = []


# ------------------------------------------------------------
# In-memory хранилища (без БД)
# ------------------------------------------------------------
# users: username -> данные пользователя и baseline.
users: dict[str, dict[str, Any]] = {}

# sessions: session_id -> данные активной сессии.
sessions: dict[str, dict[str, Any]] = {}

# event_log: последние 50 событий для дашборда.
event_log: list[dict[str, Any]] = []

# active_websockets: подключенные dashboard-сокеты.
active_websockets: set[WebSocket] = set()

# active_workspace_sockets: session_id -> websocket (для push-команд в workspace).
active_workspace_sockets: dict[str, WebSocket] = {}


# Фоновая задача UEBA-сканирования активных сессий.
ueba_background_task: asyncio.Task | None = None


# ------------------------------------------------------------
# Сервисные функции
# ------------------------------------------------------------
def append_event(event: dict[str, Any]) -> None:
    """
    Добавляет событие в event_log и ограничивает размер до 50 последних записей.
    """
    if "timestamp" not in event:
        event["timestamp"] = datetime.now().isoformat()

    event_log.append(event)

    # Храним только последние 50 событий.
    if len(event_log) > 50:
        del event_log[:-50]


async def broadcast_to_dashboards(event: dict[str, Any]) -> None:
    """
    Рассылает событие всем подключенным dashboard-клиентам.

    Если сокет отключился или отправка завершилась ошибкой,
    удаляем его из активного набора.
    """
    append_event(event)

    disconnected: set[WebSocket] = set()

    for ws in active_websockets.copy():
        try:
            await ws.send_json(event)
        except Exception:
            disconnected.add(ws)

    for ws in disconnected:
        active_websockets.discard(ws)


def _load_json_file(path: Path, default: Any) -> Any:
    """
    Безопасно загружает JSON-файл. Если файла нет или он поврежден, возвращает default.
    """
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _load_joblib_file(path: Path, default: Any = None) -> Any:
    """
    Безопасно загружает joblib-артефакт.
    Если файл отсутствует/поврежден, возвращает default.
    """
    if not path.exists():
        return default
    try:
        return joblib.load(path)
    except Exception:
        return default


def _create_session(username: str) -> str:
    """
    Создает новую пользовательскую сессию в памяти и возвращает session_id.
    """
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "username": username,
        "actions": [],
        "start_time": datetime.now(),
        "last_risk": 0.0,
    }
    return session_id


def _safe_target(value: Any) -> str:
    """
    Приводит target к удобному виду для правил UEBA.
    """
    return str(value or "").strip().lower().lstrip("/")


async def _compute_ueba_for_session(session_id: str) -> dict[str, Any]:
    """
    Общая логика вычисления UEBA-score для сессии.
    Используется и HTTP-эндпоинтом, и фоновой задачей.
    """
    session = sessions.get(session_id)
    if session is None:
        raise ValueError("Session not found")

    actions: list[dict[str, Any]] = session["actions"]
    now = datetime.now()

    session_duration_sec = max((now - session["start_time"]).total_seconds(), 0.0)

    # Warm-up режим:
    # - пока сессии < 5 секунд или действий < 2,
    #   считаем риск нейтрально-низким и НЕ гоняем модель.
    # Это убирает ложные всплески на "пустых" данных старта.
    if session_duration_sec < 5.0 or len(actions) < 2:
        score = 0.1
        session["last_risk"] = score

        payload = {
            "type": "ueba_update",
            "session_id": session_id,
            "user": session["username"],
            "score": score,
            "features": {"note": "insufficient_data"},
            "timestamp": now.isoformat(),
        }
        await broadcast_to_dashboards(payload)
        return {"score": score, "trust_level": int((1 - score) * 100)}

    # Извлекаем время действий.
    action_times = [item["timestamp"] for item in actions if isinstance(item.get("timestamp"), datetime)]
    action_times.sort()

    # clicks_per_minute:
    # - для очень коротких сессий (< 30 сек) берем окно 10 сек и масштабируем * 6;
    # - иначе используем фактическую среднюю частоту действий за сессию.
    # В любом случае гарантируем нижнюю границу 1.0 (без нулей).
    if session_duration_sec < 30.0:
        ten_seconds_ago = now - timedelta(seconds=10)
        clicks_last_10_sec = sum(
            1
            for item in actions
            if isinstance(item.get("timestamp"), datetime) and item["timestamp"] >= ten_seconds_ago
        )
        clicks_per_minute = max(float(clicks_last_10_sec) * 6.0, 1.0)
    else:
        session_duration_min_actual = max(session_duration_sec / 60.0, 1e-6)
        actual_clicks_per_minute = len(actions) / session_duration_min_actual
        clicks_per_minute = max(float(actual_clicks_per_minute), 1.0)

    # avg_time_between_clicks_sec: средняя пауза между последовательными действиями.
    # Если временных точек < 2, ставим 7.0 (типичный baseline из синтетики),
    # а не 0.0, чтобы не имитировать "сверхбыстрые клики".
    if len(action_times) >= 2:
        deltas = [
            (action_times[i] - action_times[i - 1]).total_seconds()
            for i in range(1, len(action_times))
        ]
        avg_time_between_clicks_sec = float(np.mean(deltas)) if deltas else 7.0
    else:
        avg_time_between_clicks_sec = 7.0

    # unique_pages_visited: число уникальных target.
    targets = [_safe_target(item.get("target")) for item in actions]
    unique_pages_visited = len(set(targets))

    # sensitive_pages_accessed: количество переходов в "чувствительные" разделы.
    sensitive_targets = {"admin", "export", "users"}
    sensitive_pages_accessed = sum(1 for target in targets if target in sensitive_targets)

    # session_duration_min: даем нижнюю границу 0.5 мин,
    # чтобы ранние сессии не превращались в "почти нули".
    session_duration_min = max(session_duration_sec / 60.0, 0.5)

    # data_downloaded_mb: учитываем и download, и export.
    # Для лучшего совпадения с синтетикой считаем 5 MB на действие.
    transfer_actions_count = sum(
        1
        for item in actions
        if str(item.get("action_type", "")).strip().lower() in {"download", "export"}
    )
    data_downloaded_mb = float(transfer_actions_count) * 5.0

    features = {
        "clicks_per_minute": float(clicks_per_minute),
        "avg_time_between_clicks_sec": float(avg_time_between_clicks_sec),
        "unique_pages_visited": float(unique_pages_visited),
        "sensitive_pages_accessed": float(sensitive_pages_accessed),
        "session_duration_min": float(session_duration_min),
        "data_downloaded_mb": float(data_downloaded_mb),
    }

    # Готовим DataFrame строго в том же порядке признаков, как при обучении.
    model_input = pd.DataFrame(
        [
            {
                "clicks_per_minute": features["clicks_per_minute"],
                "avg_time_between_clicks_sec": features["avg_time_between_clicks_sec"],
                "unique_pages_visited": features["unique_pages_visited"],
                "sensitive_pages_accessed": features["sensitive_pages_accessed"],
                "session_duration_min": features["session_duration_min"],
                "data_downloaded_mb": features["data_downloaded_mb"],
            }
        ]
    )

    # Прогон через пайплайн:
    # 1) StandardScaler
    # 2) decision_function IsolationForest (чем меньше, тем аномальнее)
    # 3) инверсия знака (чем больше, тем рискованнее)
    # 4) MinMaxScaler -> score в [0,1] (дополнительно clip на всякий случай)
    scaled_features = scaler.transform(model_input)
    decision_raw = isolation_forest_model.decision_function(scaled_features)[0]
    risk_raw = -decision_raw
    score = float(minmax_scaler.transform(np.array([[risk_raw]])).ravel()[0])
    score = float(np.clip(score, 0.0, 1.0))

    # Если PCA-модель доступна, вычисляем real-time 2D-проекцию live-сессии.
    # Это обеспечивает "желтые точки" из реальных признаков, без рандома.
    pca_point: dict[str, float] | None = None
    if pca_model is not None:
        projected = pca_model.transform(scaled_features)[0]
        pca_point = {"x": float(projected[0]), "y": float(projected[1])}

    session["last_risk"] = score

    event = {
        "type": "ueba_update",
        "session_id": session_id,
        "user": session["username"],
        "score": score,
        "features": features,
        "pca_point": pca_point,
        "timestamp": now.isoformat(),
    }
    await broadcast_to_dashboards(event)

    # Если риск критичный, отправляем команду terminate в workspace.
    if score > 0.7 and session_id in active_workspace_sockets:
        ws = active_workspace_sockets.get(session_id)
        if ws is not None:
            try:
                await ws.send_json(
                    {"type": "terminate", "reason": "Anomalous behavior detected"}
                )
            except Exception:
                active_workspace_sockets.pop(session_id, None)

    return {"score": score, "trust_level": int((1 - score) * 100)}


async def ueba_monitor_loop() -> None:
    """
    Фоновый цикл: каждые 3 секунды пересчитывает UEBA-score для всех активных сессий.
    """
    while True:
        for session_id in list(sessions.keys()):
            try:
                await _compute_ueba_for_session(session_id)
            except Exception:
                # Для демо достаточно "тихо" пропускать ошибочные сессии,
                # чтобы не падал весь фоновый процесс.
                continue
        await asyncio.sleep(3)


# ------------------------------------------------------------
# Startup / shutdown
# ------------------------------------------------------------
@app.on_event("startup")
async def startup_event() -> None:
    """
    Загружаем ML-артефакты и JSON-данные при старте приложения.
    Также запускаем фоновую UEBA-задачу.
    """
    global isolation_forest_model, scaler, minmax_scaler, pca_model
    global ml_metrics, feature_importance, test_scores_data, roc_curve_data, pca_projection_data
    global pr_curve_data, threshold_analysis_data
    global ueba_background_task

    isolation_forest_path = DATA_DIR / "isolation_forest.pkl"
    scaler_path = DATA_DIR / "scaler.pkl"
    minmax_scaler_path = DATA_DIR / "minmax_scaler.pkl"
    pca_model_path = DATA_DIR / "pca_model.pkl"

    isolation_forest_model = _load_joblib_file(isolation_forest_path, default=None)
    scaler = _load_joblib_file(scaler_path, default=None)
    minmax_scaler = _load_joblib_file(minmax_scaler_path, default=None)
    pca_model = _load_joblib_file(pca_model_path, default=None)

    ml_metrics = _load_json_file(DATA_DIR / "metrics.json", {})
    feature_importance = _load_json_file(DATA_DIR / "feature_importance.json", [])
    test_scores_data = _load_json_file(DATA_DIR / "test_scores.json", [])
    roc_curve_data = _load_json_file(DATA_DIR / "roc_curve_data.json", {})
    pca_projection_data = _load_json_file(DATA_DIR / "pca_projection.json", [])
    pr_curve_data = _load_json_file(DATA_DIR / "pr_curve_data.json", {})
    threshold_analysis_data = _load_json_file(DATA_DIR / "threshold_analysis.json", [])

    # Запускаем фоновый мониторинг только один раз.
    if ueba_background_task is None or ueba_background_task.done():
        ueba_background_task = asyncio.create_task(ueba_monitor_loop())


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """
    Аккуратно останавливаем фоновую задачу при завершении приложения.
    """
    global ueba_background_task
    if ueba_background_task is not None:
        ueba_background_task.cancel()
        try:
            await ueba_background_task
        except asyncio.CancelledError:
            pass


# ------------------------------------------------------------
# HTML routes
# ------------------------------------------------------------
@app.get("/")
async def root() -> RedirectResponse:
    """
    Корень приложения перенаправляет на страницу аутентификации.
    """
    return RedirectResponse(url="/static/auth.html")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page() -> HTMLResponse:
    """
    Возвращает HTML дашборда.
    """
    html = (STATIC_DIR / "dashboard.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.get("/workspace", response_class=HTMLResponse)
async def workspace_page() -> HTMLResponse:
    """
    Возвращает HTML рабочей области.
    """
    html = (STATIC_DIR / "workspace.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


# ------------------------------------------------------------
# Auth endpoints
# ------------------------------------------------------------
@app.post("/register")
async def register(request: Request) -> JSONResponse:
    """
    Регистрация пользователя и сохранение baseline-профиля.
    """
    payload = await request.json()
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))
    raw_country = str(payload.get("country", "")).strip().upper()
    user_agent = str(payload.get("user_agent", ""))
    typing_speed_ms = float(payload.get("typing_speed_ms", 200))

    if not username or not password:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "username and password are required"},
        )

    # Если фронтенд не передал country (или передал пустую строку),
    # определяем страну по IP клиента. Если не удастся, будет fallback "MD".
    if raw_country:
        country = raw_country
    else:
        client_ip = request.client.host if request.client else "127.0.0.1"
        country = get_country_from_ip(client_ip) or DEFAULT_COUNTRY

    users[username] = {
        "password": password,
        "baseline": {
            "country": country,
            "user_agent": user_agent,
            "typical_hours": list(range(7, 22)),  # 7:00-21:00
            "baseline_typing_speed_ms": typing_speed_ms,
        },
        "failed_attempts_last_hour": 0,
    }

    await broadcast_to_dashboards({"type": "register", "user": username})
    return JSONResponse(content={"status": "ok", "message": "Registered"})


@app.post("/login")
async def login(request: Request) -> JSONResponse:
    """
    Логин с адаптивной проверкой риска.
    """
    payload = await request.json()
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))
    typing_speed_ms = float(payload.get("typing_speed_ms", 200))
    user_agent = str(payload.get("user_agent", ""))

    if username not in users:
        return JSONResponse(status_code=401, content={"status": "error", "message": "Invalid credentials"})

    user = users[username]
    baseline = user["baseline"]

    # Берем IP клиента. Если по какой-то причине пусто, используем loopback.
    client_ip = request.client.host if request.client else "127.0.0.1"
    country = get_country_from_ip(client_ip)
    hour = datetime.now().hour

    # Для неверного пароля увеличиваем счетчик и сразу возвращаем 401,
    # но перед этим формируем событие (требование "в любом случае пушить событие").
    if password != user["password"]:
        user["failed_attempts_last_hour"] = int(user.get("failed_attempts_last_hour", 0)) + 1

        failed_context = {
            "ip": client_ip,
            "country": country,
            "user_agent": user_agent,
            "hour": hour,
            "typing_speed_ms": typing_speed_ms,
            "failed_attempts_last_hour": user["failed_attempts_last_hour"],
        }
        risk = calculate_auth_risk(failed_context, baseline)

        await broadcast_to_dashboards(
            {
                "type": "login_attempt",
                "user": username,
                "risk_score": risk["risk_score"],
                "decision": "invalid_password",
                "country": country,
                "triggered_rules": risk["triggered_rules"],
                "timestamp": datetime.now().isoformat(),
            }
        )
        return JSONResponse(status_code=401, content={"status": "error", "message": "Invalid credentials"})

    # На успешный пароль считаем риск адаптивной аутентификации.
    context = {
        "ip": client_ip,
        "country": country,
        "user_agent": user_agent,
        "hour": hour,
        "typing_speed_ms": typing_speed_ms,
        "failed_attempts_last_hour": int(user.get("failed_attempts_last_hour", 0)),
    }
    risk = calculate_auth_risk(context, baseline)

    # После успешной парольной проверки сбрасываем счетчик ошибок.
    user["failed_attempts_last_hour"] = 0

    event = {
        "type": "login_attempt",
        "user": username,
        "risk_score": risk["risk_score"],
        "decision": risk["decision"],
        "country": country,
        "triggered_rules": risk["triggered_rules"],
        "timestamp": datetime.now().isoformat(),
    }
    await broadcast_to_dashboards(event)

    if risk["decision"] == "block":
        return JSONResponse(
            status_code=403,
            content={
                "status": "blocked",
                "risk_score": risk["risk_score"],
                "triggered_rules": risk["triggered_rules"],
                "details": risk["details"],
            },
        )

    if risk["decision"] == "mfa_required":
        return JSONResponse(
            content={
                "need_mfa": True,
                "risk_score": risk["risk_score"],
                "triggered_rules": risk["triggered_rules"],
            }
        )

    session_id = _create_session(username)
    return JSONResponse(content={"session_id": session_id, "risk_score": risk["risk_score"]})


@app.post("/verify_mfa")
async def verify_mfa(request: Request) -> JSONResponse:
    """
    Демонстрационная MFA-проверка. Верный код всегда "1234".
    """
    payload = await request.json()
    username = str(payload.get("username", "")).strip()
    code = str(payload.get("code", ""))

    if username not in users:
        return JSONResponse(status_code=404, content={"status": "error", "message": "User not found"})

    if code != "1234":
        return JSONResponse(status_code=401, content={"status": "error", "message": "Invalid MFA code"})

    session_id = _create_session(username)
    await broadcast_to_dashboards({"type": "mfa_success", "user": username})
    return JSONResponse(content={"session_id": session_id})


# ------------------------------------------------------------
# Workspace / UEBA endpoints
# ------------------------------------------------------------
@app.post("/action")
async def action(request: Request) -> JSONResponse:
    """
    Получает действие из workspace и сохраняет его в сессию.
    """
    payload = await request.json()
    session_id = str(payload.get("session_id", "")).strip()
    action_type = str(payload.get("action_type", "")).strip()
    target = str(payload.get("target", "")).strip()

    if session_id not in sessions:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Session not found"})

    sessions[session_id]["actions"].append(
        {
            "action_type": action_type,
            "target": target,
            "timestamp": datetime.now(),
        }
    )

    return JSONResponse(content={"status": "ok"})


@app.get("/compute_ueba/{session_id}")
async def compute_ueba(session_id: str) -> JSONResponse:
    """
    Вычисляет UEBA-risk score для указанной сессии.
    """
    if session_id not in sessions:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Session not found"})

    # Проверяем, что ML-артефакты загружены.
    if isolation_forest_model is None or scaler is None or minmax_scaler is None:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "ML artifacts are not loaded"},
        )

    try:
        result = await _compute_ueba_for_session(session_id)
        return JSONResponse(content=result)
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"UEBA computation failed: {exc}"},
        )


# ------------------------------------------------------------
# API endpoints для дашборда
# ------------------------------------------------------------
@app.get("/api/ml_metrics")
async def api_ml_metrics() -> JSONResponse:
    return JSONResponse(content=ml_metrics)


@app.get("/api/feature_importance")
async def api_feature_importance() -> JSONResponse:
    return JSONResponse(content=feature_importance)


@app.get("/api/pca_data")
async def api_pca_data() -> JSONResponse:
    return JSONResponse(content=pca_projection_data)


@app.get("/api/roc_data")
async def api_roc_data() -> JSONResponse:
    return JSONResponse(content=roc_curve_data)


@app.get("/api/test_scores")
async def api_test_scores() -> JSONResponse:
    return JSONResponse(content=test_scores_data)


@app.get("/api/pr_curve")
async def api_pr_curve() -> JSONResponse:
    return JSONResponse(content=pr_curve_data)


@app.get("/api/threshold_analysis")
async def api_threshold_analysis() -> JSONResponse:
    return JSONResponse(content=threshold_analysis_data)


@app.get("/api/event_log")
async def api_event_log() -> JSONResponse:
    return JSONResponse(content=event_log[-50:])


# ------------------------------------------------------------
# WebSocket endpoints
# ------------------------------------------------------------
@app.websocket("/ws/dashboard")
async def ws_dashboard(websocket: WebSocket) -> None:
    """
    WebSocket для live-дашборда:
    - регистрируем подключение;
    - отправляем начальное состояние event_log;
    - держим соединение открытым до disconnect.
    """
    await websocket.accept()
    active_websockets.add(websocket)

    try:
        await websocket.send_json({"type": "init_event_log", "events": event_log[-50:]})
        while True:
            # Сообщения от клиента в этом демо нам не нужны,
            # но receive оставляем, чтобы отслеживать disconnect.
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_websockets.discard(websocket)
    except Exception:
        active_websockets.discard(websocket)


@app.websocket("/ws/workspace/{session_id}")
async def ws_workspace(websocket: WebSocket, session_id: str) -> None:
    """
    WebSocket для конкретной workspace-сессии.
    Через него сервер может отправить команду terminate при высоком риске.
    """
    await websocket.accept()
    active_workspace_sockets[session_id] = websocket

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_workspace_sockets.pop(session_id, None)
    except Exception:
        active_workspace_sockets.pop(session_id, None)
