from flask import Flask, request, jsonify
from transformers import pipeline

# Initialisation de l'application Flask
app = Flask(__name__)

# Chargement du pipeline de génération de texte
# Ce pipeline utilise un modèle pré-entraîné pour générer des réponses.
chatbot = pipeline("text-generation", model="gpt2")

@app.route("/chat", methods=["POST"])
def chat():
    # Récupération du message de l'utilisateur depuis la requête JSON
    user_message = request.json.get("message", "")
    
    if not user_message:
        return jsonify({"error": "Le champ 'message' est manquant."}), 400
    
    # Génération d'une réponse à partir du message de l'utilisateur
    response = chatbot(user_message, max_new_tokens=50, truncation=True)
    
    # Extraction de la réponse générée par le modèle
    generated_text = response[0]['generated_text']
    
    # Nettoyage de la réponse pour ne garder que la partie pertinente
    if user_message in generated_text:
        start_index = generated_text.find(user_message) + len(user_message)
        final_response = generated_text[start_index:].strip()
    else:
        final_response = generated_text
        
    # Retourne la réponse en format JSON
    return jsonify({"response": final_response})

if __name__ == "__main__":
    # Lancement de l'application sur le port 5000 pour le déploiement
    app.run(host='0.0.0.0', port=5000)
