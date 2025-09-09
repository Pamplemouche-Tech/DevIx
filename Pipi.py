from flask import Flask, request, jsonify
from transformers import pipeline

# Initialisation de l'application Flask
app = Flask(__name__)

# Chargement du pipeline de génération de texte
# Ce pipeline utilise un modèle pré-entraîné pour générer des réponses.
# Nous utilisons 'text-generation' pour créer une conversation.
chatbot = pipeline("text-generation", model="gpt2")

@app.route("/chat", methods=["POST"])
def chat():
    # Récupération du message de l'utilisateur depuis la requête JSON
    user_message = request.json.get("message", "")
    
    if not user_message:
        return jsonify({"error": "Le champ 'message' est manquant."}), 400
    
    # Génération d'une réponse à partir du message de l'utilisateur
    # Le paramètre `max_new_tokens` contrôle la longueur de la réponse.
    # Le paramètre `truncation=True` permet d'éviter les erreurs si le message est trop long.
    response = chatbot(user_message, max_new_tokens=50, truncation=True)
    
    # Extraction de la réponse générée par le modèle
    # Le résultat est une liste de dictionnaires, nous prenons le premier.
    generated_text = response[0]['generated_text']
    
    # Nettoyage de la réponse pour ne garder que la partie pertinente après le message de l'utilisateur
    # On cherche la première occurrence du message de l'utilisateur pour le retirer.
    # Cela permet d'éviter que le modèle ne répète le début de la conversation.
    if user_message in generated_text:
        # On ne garde que la partie de la réponse qui suit le message de l'utilisateur
        start_index = generated_text.find(user_message) + len(user_message)
        final_response = generated_text[start_index:].strip()
    else:
        # Dans de rares cas où le message n'est pas répété, on prend l'ensemble de la réponse
        final_response = generated_text
        
    # Retourne la réponse en format JSON
    return jsonify({"response": final_response})

if __name__ == "__main__":
    # Lancement de l'application
    # Host '0.0.0.0' permet à l'application d'être accessible depuis l'extérieur du conteneur
    # C'est important pour le déploiement sur des plateformes comme GitHub Pages ou des services de cloud
    app.run(host='0.0.0.0', port=5000)
