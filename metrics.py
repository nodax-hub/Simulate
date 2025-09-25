from abc import ABC, abstractmethod
import numpy as np


# === Базовый интерфейс для метрик ===
class Metric(ABC):
    @abstractmethod
    def calculate(self, v_motion, v_pump, dt, volume_total, target_density, eps=1e-9):
        """
        Должен вернуть словарь с результатами метрики
        """
        pass


# === Метрика 1. Ошибка по суммарному объему ===
class TotalVolumeError(Metric):
    def calculate(self, v_motion, v_pump, dt, volume_total, target_density, eps=1e-9):
        fact_volume = np.sum(v_pump) * dt  # литры, реально вылитые насосом
        abs_error = abs(fact_volume - volume_total)
        rel_error = abs_error / volume_total
        return {
            "fact_total_volume": fact_volume,
            "abs_total_volume_error": abs_error,
            "rel_total_volume_error": rel_error,
        }


# === Метрика 2. Ошибка равномерности внесения ===
class UniformityError(Metric):
    def calculate(self, v_motion, v_pump, dt, volume_total, target_density, eps=1e-9):
        v_motion = np.asarray(v_motion, float)
        v_pump = np.asarray(v_pump, float)

        moving = np.abs(v_motion) > eps
        instant_density = np.zeros_like(v_motion)
        instant_density[moving] = v_pump[moving] / v_motion[moving]

        mse_density = ((instant_density[moving] - target_density) ** 2).mean()
        mae_density = np.abs(instant_density[moving] - target_density).mean()

        return {
            "mse_density": mse_density,
            "mae_density": mae_density,
        }


# === Метрика 3. Переливы на остановках ===
class StopOverflowError(Metric):
    def calculate(self, v_motion, v_pump, dt, volume_total, target_density, eps=1e-9):
        v_motion = np.asarray(v_motion, float)
        v_pump = np.asarray(v_pump, float)

        stopped = np.abs(v_motion) <= eps
        stop_overflow_volume = np.sum(v_pump[stopped]) * dt
        stop_overflow_ratio = stop_overflow_volume / volume_total

        return {
            "stop_overflow_volume": stop_overflow_volume,
            "stop_overflow_ratio": stop_overflow_ratio,
        }


# === Общий класс для оценки ===
class ErrorEvaluator:
    def __init__(self, metrics, weights=None, normalization=None):
        """
        metrics: список объектов, реализующих интерфейс Metric
        weights: словарь вида { 'имя_метрики': вес }
        normalization: словарь вида { 'имя_метрики': (min_val, max_val) }
                       если не задано → без нормализации
        """
        self.metrics = metrics
        self.weights = weights or {}
        self.normalization = normalization or {}

    def evaluate(self, v_motion, v_pump, dt, volume_total, target_density, eps=1e-9):
        results = {}
        for metric in self.metrics:
            metric_result = metric.calculate(v_motion, v_pump, dt, volume_total, target_density, eps)
            results.update(metric_result)

        # добавляем интегральный скоринг
        results["integral_score"] = self.aggregate_score(results)

        return results

    def aggregate_score(self, results):
        if not self.weights:
            return None

        score = 0.0
        total_weight = 0.0

        for key, weight in self.weights.items():
            if key not in results:
                continue

            val = results[key]

            # нормализация в [0,1], если задан диапазон
            if key in self.normalization:
                min_val, max_val = self.normalization[key]
                if max_val > min_val:
                    val = (val - min_val) / (max_val - min_val)
                    val = np.clip(val, 0, 1)  # на всякий случай

            score += weight * val
            total_weight += weight

        if total_weight > 0:
            return score / total_weight

        return None