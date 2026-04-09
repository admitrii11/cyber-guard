"""
Rule-based адаптивная аутентификация (без машинного обучения).

Идея:
- На вход получаем контекст текущего логина и baseline-профиль пользователя.
- Применяем набор простых правил.
- Каждое сработавшее правило добавляет свой вклад в риск.
- По итоговому risk score выбираем решение: allow / mfa_required / block.
"""

from __future__ import annotations

import ipaddress
from typing import Any

import requests


# Домашняя страна по умолчанию (fallback), как в требовании.
DEFAULT_COUNTRY = "MD"


def get_country_from_ip(ip: str) -> str:
    """
    Определяет страну по IP через ip-api.com.

    Логика отказоустойчивости:
    1) Если IP локальный/частный (например 127.0.0.1, 192.168.x.x),
       сразу возвращаем "MD".
    2) Если сетевой запрос не удался, вернул ошибку или в ответе нет countryCode,
       также возвращаем "MD".

    Параметры:
        ip: строка с IP-адресом.

    Возвращает:
        Код страны (например "MD", "NL", "CN").
    """
    # Шаг 1: проверка IP на локальный/частный диапазон.
    # Используем модуль ipaddress: он корректно обрабатывает разные типы адресов.
    try:
        ip_obj = ipaddress.ip_address(ip)
        # is_loopback -> 127.0.0.1
        # is_private -> 192.168.x.x, 10.x.x.x, 172.16-31.x.x
        if ip_obj.is_loopback or ip_obj.is_private:
            return DEFAULT_COUNTRY
    except ValueError:
        # Если IP некорректный по формату, безопасно возвращаем fallback.
        return DEFAULT_COUNTRY

    # Шаг 2: внешний запрос к бесплатному API геолокации.
    url = f"http://ip-api.com/json/{ip}"
    try:
        response = requests.get(url, timeout=2)
        response.raise_for_status()
        payload = response.json()

        country_code = payload.get("countryCode")
        if isinstance(country_code, str) and country_code.strip():
            return country_code.upper()
    except (requests.RequestException, ValueError):
        # RequestException: сетевые ошибки/timeout/HTTP-ошибки.
        # ValueError: если JSON не распарсился.
        return DEFAULT_COUNTRY

    # Если ответ пришел, но нужного поля нет — тоже fallback.
    return DEFAULT_COUNTRY


def calculate_auth_risk(context: dict[str, Any], user_baseline: dict[str, Any]) -> dict[str, Any]:
    """
    Рассчитывает риск логина по rule-based правилам.

    Вход:
      context: контекст текущей попытки входа
      user_baseline: "нормальный профиль" пользователя

    Выход:
      {
        "risk_score": float,            # итоговый риск [0..1]
        "decision": str,                # allow / mfa_required / block
        "triggered_rules": list[str],   # список сработавших правил
        "details": dict[str, float]     # вклад каждого сработавшего правила
      }
    """
    # В этом словаре будем накапливать "вклад" сработавших правил.
    details: dict[str, float] = {}
    triggered_rules: list[str] = []
    risk_score = 0.0

    # Берем страну из context.
    # Если country не передали, пробуем определить по IP.
    # Если и это невозможно — функция get_country_from_ip вернет "MD".
    current_country = context.get("country")
    if not current_country:
        current_country = get_country_from_ip(str(context.get("ip", "")))
    baseline_country = str(user_baseline.get("country", DEFAULT_COUNTRY)).upper()
    current_country = str(current_country).upper()

    # Правило 1: новая страна.
    if current_country != baseline_country:
        details["new_country"] = 0.35
        triggered_rules.append("new_country")
        risk_score += 0.35

    # Правило 2: новый user-agent / устройство.
    baseline_user_agent = str(user_baseline.get("user_agent", ""))
    current_user_agent = str(context.get("user_agent", ""))
    if current_user_agent != baseline_user_agent:
        details["new_device"] = 0.20
        triggered_rules.append("new_device")
        risk_score += 0.20

    # Правило 3: вход в нетипичный час.
    typical_hours = user_baseline.get("typical_hours", [])
    current_hour = context.get("hour")
    if isinstance(current_hour, int) and current_hour not in typical_hours:
        details["unusual_time"] = 0.15
        triggered_rules.append("unusual_time")
        risk_score += 0.15

    # Правило 4: аномалия скорости набора.
    # "Сильно отличается" = отклонение > 50% от baseline.
    baseline_typing_speed = user_baseline.get("baseline_typing_speed_ms", 0)
    current_typing_speed = context.get("typing_speed_ms", 0)

    # Защита от деления на ноль: если baseline <= 0, просто пропускаем правило.
    if isinstance(baseline_typing_speed, (int, float)) and baseline_typing_speed > 0:
        if isinstance(current_typing_speed, (int, float)):
            deviation_ratio = abs(current_typing_speed - baseline_typing_speed) / baseline_typing_speed
            if deviation_ratio > 0.5:
                details["typing_anomaly"] = 0.25
                triggered_rules.append("typing_anomaly")
                risk_score += 0.25

    # Правило 5: признаки brute-force по числу неудачных попыток.
    failed_attempts = context.get("failed_attempts_last_hour", 0)
    if isinstance(failed_attempts, int) and failed_attempts >= 3:
        details["brute_force_suspected"] = 0.30
        triggered_rules.append("brute_force_suspected")
        risk_score += 0.30

    # Итоговый риск ограничиваем сверху 1.0.
    risk_score = min(risk_score, 1.0)

    # Правила принятия решения.
    if risk_score < 0.3:
        decision = "allow"
    elif risk_score < 0.7:
        decision = "mfa_required"
    else:
        decision = "block"

    return {
        "risk_score": round(risk_score, 2),
        "decision": decision,
        "triggered_rules": triggered_rules,
        "details": details,
    }


if __name__ == "__main__":
    # Базовый "портрет" пользователя, накопленный системой ранее.
    baseline_profile = {
        "country": "MD",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0",
        "typical_hours": list(range(8, 21)),  # 8:00-20:59
        "baseline_typing_speed_ms": 200,
    }

    # Тест 1: нормальный вход (ожидаем allow).
    normal_context = {
        "ip": "8.8.8.8",
        "country": "MD",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0",
        "hour": 14,
        "typing_speed_ms": 190,
        "failed_attempts_last_hour": 0,
    }

    # Тест 2: подозрительный вход из Китая ночью (ожидаем mfa_required или block).
    suspicious_context = {
        "ip": "1.1.1.1",
        "country": "CN",
        "user_agent": "Mozilla/5.0 (X11; Linux x86_64) Firefox/130.0",
        "hour": 3,
        "typing_speed_ms": 70,
        "failed_attempts_last_hour": 4,
    }

    normal_result = calculate_auth_risk(normal_context, baseline_profile)
    suspicious_result = calculate_auth_risk(suspicious_context, baseline_profile)

    print("=== Тест 1: Нормальный вход ===")
    print(normal_result)
    print("\n=== Тест 2: Подозрительный вход ===")
    print(suspicious_result)
