from pathlib import Path

import numpy as np

from ErrorStrategy import VolumeAbsError, DensityMSE, CompositeError
import searchers
from parse_log import Plan
from plots import plot_min_range
from use_cases import PumpingPolicy, PumpFacade


def main():
    req_norma_guess_low = 0.5
    req_norma_guess_high = 300.0
    req_norma = 20

    drone_v_min = 2.0
    drone_v_max = 10.0

    pump_min_speed = 1  # л/мин
    pump_max_speed = 13  # л/мин
    tank_volume = 40  # л

    log_file_path = Path(
        r"C:\Users\nodax\Downloads\Telegram Desktop\spraying_20250908_080420_485_24a89b48_9ecb_4956_bdb3_4d01f9de0f4a.json")

    plan = Plan.get_plan_from_log_file(log_file_path)

    # 1) чисто по объёму
    strategy_volume = VolumeAbsError()

    # 2) чисто по равномерности
    strategy_density = DensityMSE()

    # 3) комбинированная цель с весами
    strategy_combo = CompositeError([
        (strategy_volume, 1),
        (strategy_density, 0),
    ])

    permissible_error = 1 # Согласно выбранной стратегии расчёта ошибки
    strategy = strategy_combo

    policy = PumpingPolicy.NoUnderfillPolicy

    searcher = searchers.ParamSearcher()

    # start_x = req_norma_guess_low
    # stop_x = req_norma_guess_high
    # err_func = searchers.make_error_func_by_norma(
    #     plan=plan,
    #     max_drone_speed=drone_v_max,
    #     pump_min_speed=pump_min_speed,
    #     pump_max_speed=pump_max_speed,
    #     tank_volume=tank_volume,
    #     pumping_policy=policy,  # пример
    #     strategy=strategy,
    # )

    start_x = drone_v_min
    stop_x = drone_v_max
    err_func = searchers.make_error_func_by_max_drone_speed(
        plan=plan,
        norma=req_norma,
        pump_min_speed=pump_min_speed,
        pump_max_speed=pump_max_speed,
        tank_volume=tank_volume,
        pumping_policy=policy,  # пример
        strategy=strategy,
    )

    # (left_bound, right_bound), (x_minimum_err, min_error), calls = searcher.find_interval(
    #     error_func=err_func,
    #     start_x=start_x,
    #     stop_x=stop_x,
    #     eps=permissible_error
    # )
    # print("Вызовов:", calls)

    # финальная симуляция на оптимуме
    # pf = PumpFacade.from_simple_params(
    #     plan=plan,
    #     norma=req_norma,
    #     max_drone_speed=drone_v_max,
    #     pump_min_speed=pump_min_speed,
    #     pump_max_speed=pump_max_speed,
    #     tank_volume=tank_volume,
    #     pumping_policy=policy,
    # )
    #
    # pf.plot()
    # points = round(2 * calls)
    points = 300
    # plot_min_range(err_func, left_bound, right_bound, start_x, stop_x, points=points)
    plot_min_range(err_func, None, None, start_x, stop_x, points=points)

    # print(pf.volume_total, pf.total_dispensed_by_pump_plan)


if __name__ == '__main__':
    main()
