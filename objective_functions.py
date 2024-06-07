import math
import pickle
from pathlib import Path
import numpy as np

PARENT_DIR = Path(__file__).parent
MODEL_FILE = Path(PARENT_DIR, "regression_models", "time-use-health-2-0.pkl")


with open(MODEL_FILE, "rb") as model_file:
    model = pickle.load(model_file)


def calc_z1(sleep, sedentary, exercise):
    return math.sqrt(2 / 3) * math.log(sleep / math.sqrt(sedentary * exercise))


def calc_z2(sedentary, exercise):
    return (1 / math.sqrt(2)) * math.log(sedentary / exercise)


# First function to optimize
def calc_outcomes(sleep, sedentary, exercise):
    z1 = calc_z1(sleep, sedentary, exercise)
    z2 = calc_z2(sedentary, exercise)

    outcome_vals = model.predict(
        np.array([[z1, z2, z1 * z1, z1 * z2, z2 * z2]]).reshape(1, -1)
    )

    outcome_vals = outcome_vals[0]

    outcomes = {
        "Stress Level": outcome_vals[0],
        "Resting Heart Rate": outcome_vals[1],
        "Systolic Blood Pressure": outcome_vals[2],
        "Diastolic Blood Pressure": outcome_vals[3],
        "BMI": outcome_vals[4],
    }

    return outcomes


def calc_stress(solution: list) -> float:
    outcomes = calc_outcomes(solution[0], solution[1], solution[2])

    return abs(outcomes["Stress Level"])


def calc_hr(solution: list) -> float:
    outcomes = calc_outcomes(solution[0], solution[1], solution[2])

    return abs(outcomes["Resting Heart Rate"] - 80)


def calc_sbp(solution: list) -> float:
    outcomes = calc_outcomes(solution[0], solution[1], solution[2])

    return abs(outcomes["Systolic Blood Pressure"] - 115)


def calc_dbp(solution: list) -> float:
    outcomes = calc_outcomes(solution[0], solution[1], solution[2])

    return abs(outcomes["Diastolic Blood Pressure"] - 69)


def calc_bmi(solution: list) -> float:
    outcomes = calc_outcomes(solution[0], solution[1], solution[2])

    return abs(outcomes["BMI"] - 21.75)
