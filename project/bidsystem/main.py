from llama_cpp import Llama
from config import get_model_path
from agent import Agent
from moderator import DebateModerator

def main():
    # 1. モデルの読み込み
    model_path = get_model_path()
    try:
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,  # GPUに全てオフロード
            n_ctx=16384,       # コンテキストサイズ
            verbose=False,
        )
    except Exception as e:
        print(f"モデルの読み込みに失敗しました: {e}")
        return

    # 2. エージェントの初期化
    agent_names = ["A", "B", "C"]
    agents = [Agent(name=name, model=llm, system_prompt="") for name in agent_names]

    # 3. 議題の設定
    debate_topic = "人工知能は人類にとって祝福か、それとも脅威か？"
    
    # 4. モデレーターの初期化と議論の開始
    moderator = DebateModerator(agents=agents, topic=debate_topic, max_turns=20)
    moderator.run_debate()

if __name__ == "__main__":
    main()