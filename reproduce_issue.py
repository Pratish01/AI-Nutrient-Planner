import requests
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_endpoint(method, path, data=None):
    url = f"{BASE_URL}{path}"
    print(f"Testing {method} {url}...")
    try:
        if method == "POST":
            response = requests.post(url, json=data)
        else:
            response = requests.get(url)
        
        print(f"Status: {response.status_code}")
        if response.status_code == 404:
            print("❌ 404 Not Found (Expected failure before fix)")
            return False
        elif response.status_code in [200, 400, 422]:
            # 400/422 are acceptable 'success' here because it means the endpoint exists, just bad input
            print(f"✅ Endpoint exists (Status {response.status_code})")
            return True
        else:
            print(f"⚠️ Unexpected status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def run_tests():
    print("=== Reproducing 404 Issue ===")
    
    # Test 1: Register endpoint
    register_data = {
        "email": "test@example.com",
        "password": "password123",
        "name": "Test User"
    }
    
    success_reg = test_endpoint("POST", "/auth/register", register_data)
    
    # Test 2: Login endpoint
    login_data = {
        "email": "test@example.com",
        "password": "password123"
    }
    success_login = test_endpoint("POST", "/auth/login", login_data)

    if not success_reg or not success_login:
        print("\nSUMMARY: Issue Reproduced (Endpoints missing)")
    else:
        print("\nSUMMARY: Endpoints found!")

if __name__ == "__main__":
    run_tests()
