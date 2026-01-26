import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

def generate_steps_with_ollama(ingredients):
    prompt = f"""
    Create a healthy recipe using these ingredients: {ingredients}

    Rules:
    - Use minimal oil
    - Indian home-style cooking
    - Short numbered steps
    - Healthy and simple
    """

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    result = response.json()

    text = result.get("response", "")
    steps = [line.strip() for line in text.split("\n") if line.strip()]

    return steps


def generate_recipe_with_llm(ingredients):
    return {
        "title": "LLM Generated Healthy Recipe",
        "ingredients": ingredients,
        "steps": generate_steps_with_ollama(ingredients),
        "nutrition": "Low oil, balanced nutrients",
        "source": "Ollama LLM"
    }
