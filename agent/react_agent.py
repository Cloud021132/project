import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.agents import create_agent

from agent.conversation_memory import ConversationMemory
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import rag_summarize,get_weather,get_user_location,get_user_id,get_current_month,fetch_external_data,fill_context_for_report







class ReactAgent:
    def __init__(self, max_memory_turns: int = 6):
        self.memory = ConversationMemory(max_turns=max_memory_turns)
        self.agent = create_agent(
            model = chat_model,
            system_prompt = load_system_prompts(),
            tools = [rag_summarize,get_weather,get_user_location,get_user_id,get_current_month,fetch_external_data ,fill_context_for_report ],
            middleware = [monitor_tool,log_before_model,report_prompt_switch],
        )

    def execute_stream(self, query: str, session_id: str = "default"):
        input_dict = {
            "messages":[
                *self.memory.get_messages(session_id),
                {"role":"user","content":query},
            ]
        }

        answer = ""
        for chunk in self.agent.stream(input_dict,stream_mode = "values",context = {"report":False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                answer = latest_message.content.strip()
                yield answer + "\n"

        if answer:
            self.memory.add_turn(session_id, query, answer)

    def clear_memory(self, session_id: str = "default") -> None:
        self.memory.clear(session_id)



if __name__ == "__main__":
    agent = ReactAgent()

    print("开始多轮对话，输入 /clear 清空记忆，输入 /exit 退出。")
    while True:
        query = input("\n用户: ").strip()

        if query == "/exit":
            break
        if query == "/clear":
            agent.clear_memory()
            print("已清空当前会话记忆。")
            continue
        if not query:
            continue

        print("Agent: ", end="", flush=True)
        for chunk in agent.execute_stream(query):
            print(chunk, end="", flush=True)