from flask import Flask, request, jsonify
from recipe_api import fetch_recipe_from_api
from ai_engine import generate_recipe_with_llm

app = Flask(__name__)

@app.route("/recipe", methods=["POST"])
def get_recipe():
    data = request.json
    ingredients = data.get("ingredients", [])

    api_recipe = fetch_recipe_from_api(ingredients)

    if api_recipe:
        return jsonify({
            "title": api_recipe.get("title", "Healthy Recipe"),
            "ingredients": api_recipe.get("ingredients", ingredients),
            "steps": api_recipe.get("instructions", []),
            "nutrition": api_recipe.get("nutrition", {}),
            "source": "Recipe API"
        })

    # Fallback to LLM
    return jsonify(generate_recipe_with_llm(ingredients))


if __name__ == "__main__":
    app.run(debug=True)
