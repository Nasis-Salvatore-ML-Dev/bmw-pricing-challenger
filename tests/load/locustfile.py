import random
from locust import HttpUser, task, between

class BMWPricingUser(HttpUser):
    wait_time = between(1, 3)  # wait 1-3 seconds between tasks

    @task
    def predict_single(self):
        # Generate random but plausible car data
        payload = {
            "model_key": random.choice(["320d", "530i", "X5", "M3"]),
            "mileage": random.randint(5000, 200000),
            "engine_power": random.choice([184, 252, 306, 340]),
            "registration_date": f"20{random.randint(10,20)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            "fuel": random.choice(["diesel", "petrol"]),
            "paint_color": random.choice(["black", "white", "blue", "grey"]),
            "car_type": random.choice(["sedan", "suv", "coupe"]),
            "sold_at": f"20{random.randint(15,23)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            "feature_1": random.choice([True, False]),
            "feature_2": random.choice([True, False]),
            "feature_3": random.choice([True, False]),
            "feature_4": random.choice([True, False]),
            "feature_5": random.choice([True, False]),
            "feature_6": random.choice([True, False]),
            "feature_7": random.choice([True, False]),
            "feature_8": random.choice([True, False]),
        }
        self.client.post("/predict", json=payload)