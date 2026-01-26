import requests

API_KEY = "YOUR_RECIPE_API_KEY"
API_URL = "https://recipe-api.com/recipes"

def fetch_recipe_from_api(ingredients):
    try:
        response = requests.get(
            API_URL,
            headers={"X-API-Key": API_KEY},
            params={"ingredients": ",".join(ingredients)}
        )

        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                return data[0]
    except Exception as e:
        print("Recipe API Error:", e)

    return None
