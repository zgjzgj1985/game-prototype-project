import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI # <-- 使用 OpenAI 库，它与 DeepSeek 兼容
# --- 为函数计算新增导入 ---
from werkzeug.wsgi import WSGIWrapper # 确保你已安装Werkzeug
# from werkzeug.serving import run_simple # 这行通常不需要，除非你本地也用它测试
# --- 新增导入结束 ---

# 加载 .env 文件中的环境变量
load_dotenv()

app = Flask(__name__)
CORS(app) # 允许跨域请求，前端才能调用

# --- 调试信息：检查 .env 文件加载状态 ---
# 这两行调试代码可以保留在文件开头，它们会打印两次，因为Flask的debug模式会重启应用
# 如果不希望看到两次，可以注释掉它们或在部署时移除
print(f"DEBUG: .env file loaded successfully: {load_dotenv()}")
# 注意：这里我们检查的是DEEPSEEK_API_KEY
print(f"DEBUG: Attempting to get DEEPSEEK_API_KEY. Value (first 5 chars): {os.getenv('DEEPSEEK_API_KEY')[:5] if os.getenv('DEEPSEEK_API_KEY') else 'None'}")
# --- 调试代码结束 ---

# 从环境变量中获取 DeepSeek API Key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY not found in environment variables. Please set it or create a .env file.")

# 初始化 DeepSeek API 客户端 (使用 OpenAI 库的兼容模式)
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1") # <-- 这是关键的 DeepSeek API 地址

@app.route('/generate-prototype', methods=['POST'])
def generate_prototype():
    try:
        data = request.get_json()
        game_type = data.get('gameType', '通用冒险游戏')
        target_experience = data.get('targetExperience', '乐趣和挑战')

        if not game_type or not target_experience:
            return jsonify({"success": False, "message": "游戏类型和目标体验是必填项"}), 400

        # --- 核心Prompt定义 ---
        prompt = f"""
        你是一位经验丰富的游戏设计师和人工智能助手。
        请根据以下要求，为一款新游戏设计一个详细的原型，并提供核心玩法的规则。
        同时，请尝试为核心玩法循环生成一个Mermaid流程图代码。

        ### 游戏类型关键词：{game_type}
        ### 目标体验关键词：{target_experience}

        ---

        请按照以下JSON格式严格输出所有内容，不要包含任何额外说明或Markdown块以外的文本。Mermaid图表内容应放在 "mermaid_diagram" 字段中。

        ```json
        {{
            "title": "游戏原型名称：[一个吸引人的游戏名称]",
            "elevator_pitch": "游戏的核心卖点和一句话简介。",
            "core_gameplay_loop": [
                "步骤1：...",
                "步骤2：...",
                "步骤3：..."
            ],
            "key_mechanics": [
                {{
                    "name": "机制名称1",
                    "description": "详细解释机制的工作原理、玩家如何互动、重要性等。",
                    "rules": [
                        "规则点1",
                        "规则点2"
                    ]
                }},
                {{
                    "name": "机制名称2",
                    "description": "详细解释机制的工作原理、玩家如何互动、重要性等。",
                    "rules": [
                        "规则点1",
                        "规则点2"
                    ]
                }}
                // ... 可以有更多机制
            ],
            "interplay_of_mechanics": "解释这些机制如何协同作用，形成独特体验。",
            "potential_challenges": [
                "挑战/风险点1",
                "挑战/风险点2"
            ],
            "pseudocode_example": "伪代码或核心逻辑流程（可选，但推荐提供核心循环的伪代码）",
            "mermaid_diagram": "Mermaid流程图代码，例如：\\n```mermaid\\ngraph TD;\\n    A[开始]-->B(收集资源);\\n    B-->C{{资源是否充足?}};\\n    C--是-->D[建造/升级];\\n    C--否-->A;\\n    D-->E[遭遇敌人];\\n    E--胜利-->B;\\n    E--失败-->F(游戏结束);\\n```"
        }}
        ```
        """
        # --- Prompt定义结束 ---

        print(f"Generating prototype for Type: {game_type}, Experience: {target_experience}")

        # 调用 DeepSeek API (使用 OpenAI 客户端)
        completion = client.chat.completions.create(
            model="deepseek-chat", # DeepSeek 的文本生成模型名称
            messages=[
                {"role": "system", "content": "你是一位经验丰富的游戏设计师和人工智能助手。请严格按照用户要求生成JSON和Mermaid图表。"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}, # 强制AI返回JSON格式 (DeepSeek支持这个)
            temperature=0.7,
            max_tokens=2000
        )
        response_content = completion.choices[0].message.content
        print(f"AI Raw Response: {response_content}") # 打印原始响应，方便调试

        # 尝试解析JSON。即使强制JSON输出，也保留提取逻辑以增强鲁棒性
        if "```json" in response_content:
            json_start = response_content.find("```json") + len("```json")
            json_end = response_content.find("```", json_start)
            if json_end != -1:
                json_str = response_content[json_start:json_end].strip()
            else:
                json_str = response_content[json_start:].strip()
        else:
            json_str = response_content.strip()

        prototype_data = json.loads(json_str)

        return jsonify({"success": True, "data": prototype_data})

    # --- 错误处理块 ---
    except json.JSONDecodeError:
        print(f"DEBUG: AI did not return valid JSON. Raw response:\n{response_content}")
        return jsonify({"success": False, "message": f"AI返回的JSON格式不正确，请重试或优化Prompt。原始响应：{response_content}"}), 500
    except OpenAI.APIError as e: # 捕获 OpenAI (兼容 DeepSeek) 的通用 API 错误
        print(f"DeepSeek API Error: {e}")
        # 可以根据 e.type 或 e.code 区分更具体的错误，例如认证错误、限流错误等
        return jsonify({"success": False, "message": f"DeepSeek API 调用失败: {str(e)}"}), 500
    except Exception as e: # 捕获所有其他未知错误
        print(f"Error: {e}")
        return jsonify({"success": False, "message": f"服务器内部错误或API调用失败: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
