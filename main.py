from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
import requests
import os
from dotenv import load_dotenv
import uuid
import time

load_dotenv()

app = FastAPI(title="Celeste API")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurar Supabase
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# Modelo de mensaje
class Message(BaseModel):
    user_id: str
    content: str

# Modelo de respuesta
class Response(BaseModel):
    message_id: str
    response: str
    timestamp: float

@app.get("/")
def read_root():
    return {"message": "Celeste API is running"}

@app.post("/chat", response_model=Response)
async def chat(message: Message):
    try:
        # 1. Guardar mensaje del usuario en Supabase
        msg_id = str(uuid.uuid4())
        timestamp = time.time()
        
        supabase.table("messages").insert({
            "id": msg_id,
            "user_id": message.user_id,
            "content": message.content,
            "role": "user",
            "timestamp": timestamp
        }).execute()

        # 2. Obtener historial reciente del usuario (para contexto)
        history = supabase.table("messages") \
            .select("content, role") \
            .eq("user_id", message.user_id) \
            .order("timestamp", desc=True) \
            .limit(10) \
            .execute()

        # 3. Construir contexto
        context = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in reversed(history.data)
        ])

        # 4. Llamar a Ollama a través del túnel de Cloudflare
        ollama_tunnel_url = os.getenv("OLLAMA_TUNNEL_URL")
        print(f"DEBUG: OLLAMA_TUNNEL_URL = {ollama_tunnel_url}")  # Aparecerá en los logs
        if not ollama_tunnel_url:
            raise HTTPException(status_code=500, detail="OLLAMA_TUNNEL_URL no configurada")

        payload = {
            "model": "llama3.2:1b",
            "messages": [
                {
                    "role": "system",
                    "content": "Eres Celeste, una IA amigable y útil. Usa el contexto para recordar conversaciones anteriores."
                },
                {
                    "role": "user",
                    "content": f"Contexto:\n{context}\n\nNuevo mensaje: {message.content}"
                }
            ],
            "stream": False
        }

        try:
            response_ollama = requests.post(f"{ollama_tunnel_url}/api/chat", json=payload)
            response_ollama.raise_for_status()
            ollama_response_data = response_ollama.json()
            respuesta_texto = ollama_response_data['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"Error conectando con Ollama: {e}")
            raise HTTPException(status_code=503, detail="No se pudo contactar con el modelo de IA local.")

        # 5. Guardar respuesta en Supabase
        supabase.table("messages").insert({
            "id": str(uuid.uuid4()),
            "user_id": message.user_id,
            "content": respuesta_texto,
            "role": "assistant",
            "timestamp": time.time()
        }).execute()

        # 6. Devolver respuesta
        return Response(
            message_id=msg_id,
            response=respuesta_texto,
            timestamp=timestamp
        )
    except Exception as e:
        print(f"Error general: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{user_id}")
async def get_history(user_id: str, limit: int = 50):
    """Obtiene el historial de conversación de un usuario"""
    history = supabase.table("messages") \
        .select("content, role, timestamp") \
        .eq("user_id", user_id) \
        .order("timestamp", desc=True) \
        .limit(limit) \
        .execute()
    return {"history": history.data}