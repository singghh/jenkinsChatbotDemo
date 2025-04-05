import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import sqlite3
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from fuzzywuzzy import process
import pickle
import numpy as np

app = FastAPI()

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect("jenkins_cache.db")
    cursor = conn.cursor()
    # Plugins table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS plugins (
            name TEXT PRIMARY KEY,
            description TEXT
        )
    """)
    plugins = [
        ("git", "Git plugin for source code management in Jenkins"),
        ("blueocean", "Blue Ocean plugin for enhanced pipeline visualization"),
        ("maven-plugin", "Maven plugin for building Java projects"),
        ("docker-workflow", "Docker plugin for containerized builds"),  # Changed from "docker"
        ("slack", "Slack plugin for sending build notifications"),
        ("workflow-aggregator", "Pipeline plugin for defining CI/CD workflows"),
        ("github", "GitHub plugin for integrating with GitHub repositories"),
        ("junit", "JUnit plugin for publishing test results"),
        ("email-ext", "Email Extension plugin for advanced email notifications"),
        ("credentials", "Credentials plugin for managing secure credentials"),
        ("sonar", "SonarQube plugin for code quality analysis"),
        ("artifactory", "Artifactory plugin for managing build artifacts"),
    ]
    cursor.executemany("INSERT OR IGNORE INTO plugins (name, description) VALUES (?, ?)", plugins)

    # Docs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS docs (
            topic TEXT PRIMARY KEY,
            link TEXT
        )
    """)
    docs = [
        ("pipelines", "https://www.jenkins.io/doc/book/pipeline/"),
        ("jobs", "https://www.jenkins.io/doc/book/using/"),
        ("nodes", "https://www.jenkins.io/doc/book/managing/nodes/"),
        ("credentials", "https://www.jenkins.io/doc/book/using/managing-credentials/"),
        ("multibranch", "https://www.jenkins.io/doc/book/pipeline/multibranch/"),
        ("security", "https://www.jenkins.io/doc/book/security/"),
        ("plugins", "https://www.jenkins.io/doc/book/managing/plugins/"),
    ]
    cursor.executemany("INSERT OR IGNORE INTO docs (topic, link) VALUES (?, ?)", docs)

    # Snippets table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS snippets (
            topic TEXT PRIMARY KEY,
            snippet TEXT
        )
    """)
    snippets = [
        ("pipeline", "pipeline { agent any stages { stage('Build') { steps { echo 'Building...' } } } }"),
        ("git", "git url: 'https://github.com/user/repo.git', branch: 'main'"),
        ("slack", "slackSend channel: '#builds', message: 'Build completed!'"),
    ]
    cursor.executemany("INSERT OR IGNORE INTO snippets (topic, snippet) VALUES (?, ?)", snippets)

    conn.commit()
    return conn

# Global database connection
db = init_db()

# Load fine-tuned DistilBert model and label encoder
tokenizer = DistilBertTokenizer.from_pretrained("./distilbert_finetuned")
model = DistilBertForSequenceClassification.from_pretrained("./distilbert_finetuned")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Fuzzy matching for plugins, docs, and snippets
def get_closest_match(query_name, table, column):
    cursor = db.cursor()
    cursor.execute(f"SELECT {column} FROM {table}")
    names = [row[0] for row in cursor.fetchall()]
    closest_match, score = process.extractOne(query_name, names)
    if score > 80:
        return closest_match
    return None

# Intent classification with fine-tuned DistilBERT
def classify_intent(query):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    intent = label_encoder.inverse_transform([predicted_class])[0]
    return intent

# Process query based on intent
def process_query(query, intent):
    query_lower = query.lower()
    cursor = db.cursor()
    
    if intent == "plugin_query":
        plugin_name = query_lower.replace("show me", "").replace("tell me about", "").replace("what is the", "").replace("describe the", "").replace("plugin", "").strip()
        cursor.execute("SELECT description FROM plugins WHERE name = ?", (plugin_name,))
        result = cursor.fetchone()
        if result:
            return f"{result[0]} (Learn more: https://updates.jenkins.io/download/plugins/{plugin_name}/)"
        closest = get_closest_match(plugin_name, "plugins", "name")
        if closest:
            cursor.execute("SELECT description FROM plugins WHERE name = ?", (closest,))
            result = cursor.fetchone()
            return f"Did you mean '{closest}'? {result[0]} (Learn more: https://updates.jenkins.io/download/plugins/{closest}/)"
        return f"Sorry, I couldn't find the '{plugin_name}' plugin. Try searching on updates.jenkins.io."
    
    elif intent == "doc_request":
        topic = None
        for t in ["pipelines", "jobs", "nodes", "credentials", "multibranch", "security", "plugins"]:
            if t in query_lower:
                topic = t
                break
        if not topic:
            topic = query_lower.replace("i need docs for", "").replace("documentation for", "").replace("docs on", "").strip()
        cursor.execute("SELECT link FROM docs WHERE topic = ?", (topic,))
        result = cursor.fetchone()
        if result:
            return f"Here’s the documentation for {topic}: {result[0]}"
        closest = get_closest_match(topic, "docs", "topic")
        if closest:
            cursor.execute("SELECT link FROM docs WHERE topic = ?", (closest,))
            result = cursor.fetchone()
            return f"Did you mean '{closest}'? Here’s the documentation: {result[0]}"
        return f"Sorry, I couldn’t find documentation for {topic}. Try searching on jenkins.io/doc."
    
    elif intent == "snippet_request":
        topic = None
        for t in ["pipeline", "git", "slack"]:
            if t in query_lower:
                topic = t
                break
        if not topic:
            topic = query_lower.replace("give me a", "").replace("show me a", "").replace("example", "").replace("snippet", "").strip()
        cursor.execute("SELECT snippet FROM snippets WHERE topic = ?", (topic,))
        result = cursor.fetchone()
        if result:
            return f"Here’s a code snippet for {topic}:\n```groovy\n{result[0]}\n```"
        closest = get_closest_match(topic, "snippets", "topic")
        if closest:
            cursor.execute("SELECT snippet FROM snippets WHERE topic = ?", (closest,))
            result = cursor.fetchone()
            return f"Did you mean '{closest}'? Here’s a snippet:\n```groovy\n{result[0]}\n```"
        return f"No snippet found for {topic}."
    
    return "I didn't understand your query. Try asking about a plugin, documentation, or snippet."

# HTML for WebSocket client
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Jenkins Chatbot Demo with DistilBERT</title>
    </head>
    <body>
        <h1>Jenkins Chatbot Demo</h1>
        <input type="text" id="message" placeholder="Type a query (e.g., Show me Git plugin)" />
        <button onclick="sendMessage()">Send</button>
        <p id="response"></p>
<script>
    var ws = new WebSocket("wss://jenkinschatbotdemo.onrender.com/ws");
    ws.onmessage = function(event) {
        document.getElementById("response").innerText = event.data;
    };
    function sendMessage() {
        var message = document.getElementById("message").value;
        ws.send(message);
    }
</script>
    </body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        intent = classify_intent(data)
        response = process_query(data, intent)
        await websocket.send_text(response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
