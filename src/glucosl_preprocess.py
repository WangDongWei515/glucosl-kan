import numpy as np


CLIP = 3.0


class Preprocessor:
    # Training-data medians and standardization; person variables use one value per patient (protocol G1, 4.1).
    # Kept free of PyTorch so that the comparators can run in a process without it.
    def __init__(self, person, meal):
        self.person, self.meal = list(person), list(meal)

    def fit(self, frame, groups):
        patients = frame[self.person].astype(float).groupby(np.asarray(groups)).first()
        self.person_median = patients.median()
        patients = patients.fillna(self.person_median)
        self.person_mean, self.person_sd = patients.mean(), patients.std().replace(0, 1)
        meals = frame[self.meal].astype(float)
        self.meal_median = meals.median()
        meals = meals.fillna(self.meal_median)
        self.meal_mean, self.meal_sd = meals.mean(), meals.std().replace(0, 1)
        return self

    def transform(self, frame, clip=False):
        person = (frame[self.person].astype(float).fillna(self.person_median) - self.person_mean) / self.person_sd
        meal = (frame[self.meal].astype(float).fillna(self.meal_median) - self.meal_mean) / self.meal_sd
        if clip:
            person, meal = person.clip(-CLIP, CLIP), meal.clip(-CLIP, CLIP)
        return person.to_numpy(dtype=np.float32), meal.to_numpy(dtype=np.float32)
