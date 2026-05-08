from graph import app
from langchain_core.messages import HumanMessage

def run_rag_system():
    print("="*60)
    print("工业级多智能体 RAG 问答系统 升级完成！")
    print("="*60)
    
    while True:
        user_input = input("\n请输入你的问题（输入 q 退出）：")
        if user_input.lower() == 'q':
            print("👋 系统退出，再见！")
            break
        
        print("\n" + "-"*60)
        # 调用升级后的LangGraph工作流
        result = app.invoke({
            "messages": [HumanMessage(content=user_input)]
        })
        
        print("\n" + "="*60)
        print("最终回答：")
        print(result["final_answer"])
        print("="*60)

if __name__ == "__main__":
    run_rag_system()