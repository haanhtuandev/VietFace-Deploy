# face_analysis/app/components/emoti_chat/chat_ui.py

import streamlit as st 
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from src.services.chat_service import ChatService
from app.state.managers.state_manager import StateManager

st.cache_data.clear()
st.cache_resource.clear()
@dataclass
class ChatUIComponents:
    @staticmethod
    def message_container(message_html: str) -> str:
        return (
            f'<div class="chat-container" id="chat-container">'
            f'{message_html}'
            f'</div>'
        )

    @staticmethod
    def emotion_badge(emotion: str) -> str:
        colors = {
            'happy': ('#4CAF50', '#E8F5E9'),     
            'sad': ('#5C6BC0', '#E8EAF6'),       
            'angry': ('#F44336', '#FFEBEE'),     
            'surprised': ('#FF9800', '#FFF3E0'), 
            'fear': ('#7E57C2', '#EDE7F6'),   
            'disgusted': ('#795548', '#EFEBE9'), 
            'neutral': ('#78909C', '#ECEFF1')    
        }
        
        emoji_map = {
            'happy': '😊', 'sad': '😢', 'angry': '😠',
            'surprised': '😮', 'fear': '😨',
            'disgusted': '😖', 'neutral': '😐'
        }
        
        if not emotion or not isinstance(emotion, str):
            return ""
        emotion = emotion.lower()
        if emotion not in colors:
            return ""
            
        color, bg = colors[emotion]
        emoji = emoji_map[emotion]
        return f'<div class="emotion-badge" style="color:{color};background-color:{bg};border:1px solid {color}40">{emoji} {emotion}</div>'

    @staticmethod
    def message_bubble(content: str, is_user: bool, emotion: Optional[str], timestamp: str) -> str:
        message_class = "chat-message-user" if is_user else "chat-message-bot"
        wrapper_class = "message-wrapper-right" if is_user else "message-wrapper-left"
        emotion_display = ChatUIComponents.emotion_badge(emotion) if emotion else ""
        
        return (
            f'<div class="{wrapper_class}">'  # Add wrapper div
            f'<div class="{message_class}">'
            f'<div class="message-content">{content}</div>'
            f'{emotion_display}'
            f'<div class="timestamp">{timestamp}</div>'
            f'</div>'
            f'</div>'
        )

class ChatUI:
    def __init__(self, state_manager: StateManager, chat_service: ChatService):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        self.state_manager = state_manager
        self.chat_service = chat_service
        self.components = ChatUIComponents()
        
        # if not hasattr(st.session_state, 'chat_ui_initialized'):
        #     self._init_session_state()
        #     self._init_styles()
        #     st.session_state.chat_ui_initialized = True
        
        self._init_session_state()
        self._init_styles()

    def _init_session_state(self) -> None:
        if 'chat_input' not in st.session_state:
            st.session_state.chat_input = ''
            
    def _init_styles(self) -> None:
        st.markdown("""
            <style>
                .chat-container {
                    height: 600px;
                    width: 100%;
                    max-width: 900px;
                    overflow-y: auto;
                    padding: 24px;
                    background: #1a1a1a;
                    border-radius: 24px;
                    margin-bottom: 20px;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
                    scrollbar-width: thin;
                    position: relative;
                    backdrop-filter: blur(10px);
                }

                .chat-container::-webkit-scrollbar {
                    width: 6px;
                }

                .chat-container::-webkit-scrollbar-track {
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 3px;
                }

                .chat-container::-webkit-scrollbar-thumb {
                    background: rgba(255, 255, 255, 0.2);
                    border-radius: 3px;
                }

                .chat-container::-webkit-scrollbar-thumb:hover {
                    background: rgba(255, 255, 255, 0.3);
                }

                .message-wrapper-left, .message-wrapper-right {
                    width: 100%;
                    display: flex;
                    margin: 16px 0;
                    clear: both;
                }

                .message-wrapper-right {
                    justify-content: flex-end;
                }

                .chat-message-user {
                    background: linear-gradient(135deg, #2c2ec2, #1e298b);
                    color: white;
                    padding: 16px 20px;
                    border-radius: 20px;
                    border-bottom-right-radius: 4px;
                    max-width: 60%;
                    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.2);
                    animation: slideInRight 0.4s cubic-bezier(0.16, 1, 0.3, 1);
                    position: relative;
                    backdrop-filter: blur(10px);
                }

                .chat-message-bot {
                    background: rgba(255, 255, 255, 0.05);
                    color: #fff;
                    padding: 16px 20px;
                    border-radius: 20px;
                    border-bottom-left-radius: 4px;
                    max-width: 60%;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
                    animation: slideInLeft 0.4s cubic-bezier(0.16, 1, 0.3, 1);
                    position: relative;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }

                .message-content {
                    line-height: 1.6;
                    font-size: 15px;
                    letter-spacing: 0.3px;
                    word-wrap: break-word;
                    font-weight: 400;
                }

                .emotion-badge {
                    font-size: 12px;
                    padding: 6px 12px;
                    margin-top: 8px;
                    border-radius: 12px;
                    font-weight: 500;
                    display: inline-flex;
                    align-items: center;
                    gap: 4px;
                    animation: fadeIn 0.4s ease-out;
                    background: rgba(255, 255, 255, 0.1);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    backdrop-filter: blur(4px);
                }

                .timestamp {
                    font-size: 11px;
                    margin-top: 8px;
                    opacity: 0.7;
                    font-weight: 500;
                    letter-spacing: 0.3px;
                }

                .chat-message-user .timestamp {
                    color: rgba(255, 255, 255, 0.9);
                    text-align: right;
                }

                .chat-message-bot .timestamp {
                    color: rgba(255, 255, 255, 0.7);
                }

                @keyframes slideInRight {
                    from {
                        transform: translateX(30px);
                        opacity: 0;
                    }
                    to {
                        transform: translateX(0);
                        opacity: 1;
                    }
                }

                @keyframes slideInLeft {
                    from {
                        transform: translateX(-30px);
                        opacity: 0;
                    }
                    to {
                        transform: translateX(0);
                        opacity: 1;
                    }
                }

                @keyframes fadeIn {
                    from {
                        transform: translateY(5px);
                        opacity: 0;
                    }
                    to {
                        transform: translateY(0);
                        opacity: 1;
                    }
                }

                @media (max-width: 768px) {
                    .chat-container {
                        height: 80vh;
                        padding: 16px;
                        border-radius: 20px;
                    }
                    
                    .chat-message-user,
                    .chat-message-bot {
                        max-width: 80%;
                        padding: 14px 16px;
                    }
                    
                    .message-content {
                        font-size: 14px;
                    }
                }
            </style>
        """, unsafe_allow_html=True)
            
    def _render_header(self) -> None:
        st.markdown("""
            <div style='margin-bottom: 25px;'>
                <h3 style='color: #1976D2; margin: 0; display: flex; align-items: center; gap: 8px;'>
                    💬 <span style='background: linear-gradient(135deg, #0D47A1, #1976D2);
                                  -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                        EmotiChat
                    </span>
                </h3>
            </div>
        """, unsafe_allow_html=True)
        
        current_emotion = self.state_manager.get_chat_state().current_emotion
        if current_emotion:
            st.markdown(f"""
                <div style='
                    padding: 12px 16px;
                    background: linear-gradient(135deg, #0D47A1, #1976D2);
                    color: white;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    font-size: 15px;
                    letter-spacing: 0.3px;
                    box-shadow: 0 2px 8px rgba(25,118,210,0.2);
                '>
                    <span style='opacity: 0.9;'>Current Emotion:</span>
                    {self.components.emotion_badge(current_emotion)}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("👋 Upload or capture an image to start emotion-aware chat!")

    def _render_messages(self) -> None:
        try:
            messages = self.chat_service.get_history()
            messages_html = "".join(
                self.components.message_bubble(
                    msg['content'],
                    msg.get('is_user', False),
                    msg.get('emotion'),
                    datetime.fromisoformat(msg['timestamp']).strftime("%H:%M")
                ) for msg in messages
            )
            
            st.markdown(
                self.components.message_container(messages_html) + 
                '<script>setTimeout(function(){document.getElementById("chat-container").scrollTop=1e6},100)</script>',
                unsafe_allow_html=True
            )
        except Exception as e:
            self.logger.error(f"Error rendering messages: {str(e)}")
            st.error("Chat error. Please refresh.")

    def _handle_input(self) -> None:
        message = st.session_state.get('chat_input', '').strip()
        if not message:
            return
            
        if len(message) > 500:
            st.error("Message too long! Please keep it under 500 characters.")
            return
            
        current_emotion = self.state_manager.get_chat_state().current_emotion
        self.logger.info(f"Processing message with emotion: {current_emotion}")
        
        try:
            # Add user message & generate response
            self.chat_service.add_message(message, True, current_emotion)
            response = self.chat_service.generate_response(message, current_emotion)
            self.chat_service.add_message(response, False, current_emotion)
            
            # Clear input
            st.session_state.chat_input = ''
            
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            self.chat_service.add_message(
                "I'm having trouble right now. Please try again.",
                False
            )

    def render(self) -> None:
        with st.container():
            self._render_header()
            self._render_messages()
            
            col1, col2 = st.columns([6,1])
            with col1:
                st.text_input(
                    "Message",
                    key="chat_input",
                    placeholder="Type your message here...",
                    on_change=self._handle_input,
                    label_visibility="collapsed"
                )
            with col2:
                if st.button("Clear", key="clear_chat"):
                    self.chat_service.clear_history()

    def _clean_message_content(self, content: str) -> str:
        """Clean and escape message content for safe rendering"""
        # Remove code block markers that might cause rendering issues
        content = content.replace('```', '')
        # Escape HTML special characters except emojis
        content = (
            content
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#39;')
        )
        return content

    