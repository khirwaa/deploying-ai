import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
from agents import get_graph
# from utils.logger import get_logger


load_dotenv('../.secrets')


llm = get_graph()

def sanitize_input(user_input: str) -> str:
    return user_input.strip()

def sanitize_chat_history(history: list[dict]):
    sanitized_history = []
    for msg in history:
        if 'role' in msg and 'content' in msg:
            sanitized_history.append({
                'role': msg['role'],
                'content': msg['content']
            })
    return sanitized_history

def extract_tool_calls(history: list) -> list[dict]:
    """Extract all tool call info from a LangChain message history."""
    results = []

    for msg in history:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tool_call in msg.tool_calls:
                results.append({
                    "tool_name": tool_call["name"],
                    "tool_args": tool_call["args"],
                    "tool_call_id": tool_call["id"],
                })

    return results

def main():
    if not os.environ.get("API_GATEWAY_KEY"):
        raise ValueError("Missing API_GATEWAY_KEY environment variable")

    def chat_processing(message: str, history: list[dict]):
        langchain_messages = []
        n = 0
        for msg in history:
            if msg['role'] == 'user':
                langchain_messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                langchain_messages.append(AIMessage(content=msg['content']))
                n += 1
        langchain_messages.append(HumanMessage(content=message))

        state = {
            "messages": langchain_messages,
            "llm_calls": n
        }
        try:
            response = llm.invoke(state)
            extracted_tool_calls = extract_tool_calls(response['messages'])
            # _logs.info(f"Extracted tool calls: {extracted_tool_calls}")
            return response['messages'][len(response['messages']) - 1].content
        except Exception as e: # except GraphRecursionError
            print(f"Error invoking LLM: {e}")
            # _logs.error(f"Error invoking LLM: {e}")
            return "Sorry, there was an error processing your request. Please try again later."


        
    chat = gr.ChatInterface(
        fn=chat_processing,
        type="messages"
    )
    return chat

if __name__ == "__main__":
    chat = main()
    chat.launch()