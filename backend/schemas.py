from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

class Token(BaseModel):
    access_token: str
    token_type: str

class UserCreate(BaseModel):
    username: str   
    password: str

class UserLogin(BaseModel): 
    username: str
    password: str

class ChatCreate(BaseModel):
    title: str

class MessageCreate(BaseModel):
    role: str
    text: str
    stage_id: Optional[int]  # ✅ Track which GPT stage handled this

class ChatResponse(BaseModel):
    id: int
    title: str
    timestamp: datetime

class MessageResponse(BaseModel):
    id: int
    chat_id: int
    role: str
    text: str
    stage_id: Optional[int]  # ✅ Keep track of GPT stage
    timestamp: datetime

    class Config:
        orm_mode = True
