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
        <style>
            body {
                display: flex;
                justify-content: center;
                align-items: center;
                flex-direction: column;
                height: 100vh;
                margin: 0;
                font-family: Arial, sans-serif;
                background-color: #f4f4f9;
            }
            .header {
                text-align: center;
                margin-bottom: 20px;
            }
            .header h1 {
                font-size: 24px;
                color: #333;
                margin-bottom: 10px;
            }
            .header p {
                font-size: 14px;
                color: #555;
                margin: 0;
            }
            .chat-container {
                text-align: center;
                background: #ffffff;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                width: 400px;
                display: flex;
                flex-direction: column;
                height: 80vh;
            }
            .chat-container img {
                width: 100px;
                margin-bottom: 20px;
            }
            .chat-box {
                flex: 1;
                overflow-y: auto;
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 10px;
                margin-bottom: 10px;
                background-color: #f9f9f9;
            }
            .chat-box .message {
                margin: 5px 0;
                padding: 10px;
                border-radius: 5px;
            }
            .chat-box .user {
                background-color: #007bff;
                color: white;
                text-align: right;
            }
            .chat-box .ai {
                background-color: #e9ecef;
                color: #333;
                text-align: left;
            }
            .chat-container input {
                width: calc(100% - 90px);
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
            .chat-container button {
                padding: 10px 20px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                margin-left: 10px;
            }
            .chat-container button:hover {
                background-color: #0056b3;
            }
            .loading {
                display: none;
                margin-top: 10px;
            }
            .box {
                display: flex;
                flex-direction: row;
                align-items: center;
                justify-content: center;
                width: 100%;
                height: 100%;
                gap: 20px;
            }
        </style>
    </head>
    <body>
        <div class="box">
        <div class="header">
            <h1>Jenkins Chatbot Demo (Limited Dataset of 30 Rows)</h1>
            <p>This is for showcasing my skills, dedication, and interest in developing innovative solutions.</p>
            <b>Please be patient; the chatbot will respond as it is deployed on Render.</b>
            <p>GitHub Repository: <a href="https://github.com/singghh/jenkinsChatbotDemo" target="_blank">https://github.com/singghh/jenkinsChatbotDemo</a></p>
        </div>
        <div class="chat-container">
            <img src="https://www.jenkins.io/images/logos/jenkins/jenkins.svg" alt="Jenkins Logo" />
            
            <div class="chat-box" id="chat-box">
                <div class="message ai">Hi there! I'm Jenkins Chat Genius, your AI assistant. How can I help you today?</div>
            </div>
            <div style="display: flex;">
                <input type="text" id="message" placeholder="Type your query here..." />
                <button onclick="sendMessage()">Send</button>
            </div>
            <div class="loading" id="loading">Loading...</div>
        </div>
        </div>
        <script>
            var ws = new WebSocket("wss://jenkinschatbotdemo.onrender.com/ws");
            ws.onopen = function(event) {
                console.log("WebSocket connection established.");
            };
            ws.onmessage = function(event) {
                document.getElementById("loading").style.display = "none";
                var chatBox = document.getElementById("chat-box");
                var aiMessage = document.createElement("div");
                aiMessage.className = "message ai";
                aiMessage.innerText = event.data;
                chatBox.appendChild(aiMessage);
                chatBox.scrollTop = chatBox.scrollHeight;
            };
            function sendMessage() {
                var message = document.getElementById("message").value;
                if (message.trim() === "") return;
                var chatBox = document.getElementById("chat-box");
                var userMessage = document.createElement("div");
                userMessage.className = "message user";
                userMessage.innerText = message;
                chatBox.appendChild(userMessage);
                chatBox.scrollTop = chatBox.scrollHeight;
                document.getElementById("loading").style.display = "block";
                document.getElementById("message").value = "";
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
