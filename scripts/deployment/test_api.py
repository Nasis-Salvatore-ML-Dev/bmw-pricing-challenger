"""
BMW Pricing API - Test Script

Tests all API endpoints locally or on Cloud Run
"""

import json
import time

import requests

# Configuration
BASE_URL = "http://localhost:8000"  # Change to Cloud Run URL after deployment


def test_root():
    """Test root endpoint"""
    print("\n" + "=" * 60)
    print("TEST 1: Root Endpoint")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/", timeout=5)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200
    print("✅ PASSED")


def test_health():
    """Test health check"""
    print("\n" + "=" * 60)
    print("TEST 2: Health Check")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/health", timeout=5)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✅ PASSED")


def test_single_prediction():
    """Test single prediction"""
    print("\n" + "=" * 60)
    print("TEST 3: Single Prediction")
    print("=" * 60)

    payload = {
        "model_key": "320d",
        "mileage": 120000,
        "engine_power": 184,
        "registration_date": "2015-03-01",
        "fuel": "diesel",
        "paint_color": "black",
        "car_type": "sedan",
        "sold_at": "2020-06-15",
        # Binary features (defaults to False if not specified)
        "feature_1": False,
        "feature_2": False,
        "feature_3": False,
        "feature_4": False,
        "feature_5": False,
        "feature_6": False,
        "feature_7": False,
        "feature_8": False,
    }

    print(f"Request: {json.dumps(payload, indent=2)}")

    start = time.time()
    response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=5)
    latency = (time.time() - start) * 1000

    print(f"\nStatus: {response.status_code}")
    print(f"Latency: {latency:.1f}ms")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200
    assert "predicted_price" in response.json()
    assert latency < 200  # Should be < 200ms
    print(f"✅ PASSED (latency: {latency:.1f}ms)")


def test_batch_prediction():
    """Test batch prediction"""
    print("\n" + "=" * 60)
    print("TEST 4: Batch Prediction (3 cars)")
    print("=" * 60)

    payload = {
        "cars": [
            {
                "model_key": "320d",
                "mileage": 120000,
                "engine_power": 184,
                "registration_date": "2015-03-01",
                "fuel": "diesel",
                "paint_color": "black",
                "car_type": "sedan",
                "sold_at": "2020-06-15",
                "feature_1": False,
                "feature_2": False,
                "feature_3": False,
                "feature_4": False,
                "feature_5": False,
                "feature_6": False,
                "feature_7": False,
                "feature_8": False,
            },
            {
                "model_key": "530i",
                "mileage": 80000,
                "engine_power": 252,
                "registration_date": "2017-09-15",
                "fuel": "petrol",
                "paint_color": "white",
                "car_type": "sedan",
                "sold_at": "2021-03-20",
                "feature_1": True,
                "feature_2": False,
                "feature_3": False,
                "feature_4": False,
                "feature_5": False,
                "feature_6": False,
                "feature_7": False,
                "feature_8": False,
            },
            {
                "model_key": "X5_xDrive30d",
                "mileage": 95000,
                "engine_power": 265,
                "registration_date": "2018-01-10",
                "fuel": "diesel",
                "paint_color": "blue",
                "car_type": "suv",
                "sold_at": "2022-08-05",
                "feature_1": False,
                "feature_2": True,
                "feature_3": False,
                "feature_4": False,
                "feature_5": False,
                "feature_6": False,
                "feature_7": False,
                "feature_8": False,
            },
        ]
    }

    start = time.time()
    response = requests.post(f"{BASE_URL}/predict/batch", json=payload, timeout=5)
    latency = (time.time() - start) * 1000

    print(f"Status: {response.status_code}")
    print(f"Latency: {latency:.1f}ms")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200
    assert response.json()["total_cars"] == 3
    print(f"✅ PASSED (latency: {latency:.1f}ms)")


def test_metrics():
    """Test metrics endpoint"""
    print("\n" + "=" * 60)
    print("TEST 5: Metrics Endpoint")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/metrics", timeout=5)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200
    print("✅ PASSED")


def test_error_handling():
    """Test error handling"""
    print("\n" + "=" * 60)
    print("TEST 6: Error Handling (Invalid Input)")
    print("=" * 60)

    # Missing required field
    payload = {
        "model_key": "320d",
        "mileage": 120000,
        # Missing other required fields
    }

    response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=5)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 422  # Validation error
    print("✅ PASSED (error handled correctly)")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("BMW PRICING API - AUTOMATED TESTS")
    print("=" * 60)
    print(f"Testing: {BASE_URL}")

    tests = [
        ("Root Endpoint", test_root),
        ("Health Check", test_health),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Metrics", test_metrics),
        ("Error Handling", test_error_handling),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {failed} TESTS FAILED")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
