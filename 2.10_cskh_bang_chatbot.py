import ollama
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import json


def he_thong_chuyen_complaint_thanh_so_tu_1_den_10(user_input):
    prompt = f"""
    Analyze the customer message: "{user_input}"
    Score it from 1-10 on:
    1. urgency (1: normal, 10: immediate action)
    2. complexity (1: simple FAQ, 10: technical issue)
    Return ONLY a JSON object: {{"urgency": x, "complexity": y}}
    """
    
    try:
        response = ollama.chat(model='llama3', messages=[
            {'role': 'user', 'content': prompt},
        ])
        
        content = response['message']['content']
        start = content.find('{')
        end = content.rfind('}') + 1
        data = json.loads(content[start:end])
        
        return float(data['urgency']), float(data['complexity'])
    except Exception as e:
        print(f"Lỗi khi gọi Local AI: {e}")
        return 5.0, 5.0


urgency = ctrl.Antecedent(np.arange(0, 11, 1), 'urgency')
complexity = ctrl.Antecedent(np.arange(0, 11, 1), 'complexity')
priority = ctrl.Consequent(np.arange(0, 11, 1), 'priority')

urgency.automf(3, names=['low', 'medium', 'high'])
complexity.automf(3, names=['low', 'medium', 'high'])

priority['bot'] = fuzz.trimf(priority.universe, [0, 0, 4])
priority['staff'] = fuzz.trimf(priority.universe, [3, 6, 8])
priority['expert'] = fuzz.trimf(priority.universe, [7, 10, 10])

rule1 = ctrl.Rule(urgency['high'] | complexity['high'], priority['expert'])
rule2 = ctrl.Rule(urgency['medium'], priority['staff'])
rule3 = ctrl.Rule(urgency['low'] & complexity['low'], priority['bot'])

chatbot_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
hethong_quyetdinh = ctrl.ControlSystemSimulation(chatbot_ctrl)


tin_nhan = input("Nhập tin nhắn khách hàng: ")
u_score, c_score = he_thong_chuyen_complaint_thanh_so_tu_1_den_10(tin_nhan)

hethong_quyetdinh.input['urgency'] = u_score
hethong_quyetdinh.input['complexity'] = c_score
hethong_quyetdinh.compute()

final_score = hethong_quyetdinh.output['priority']

print("\n" + "="*30)
print(f"Llama 3 chấm điểm: Gấp({u_score}), Phức tạp({c_score})")
print(f"Kết quả Logic mờ: {round(final_score, 2)}/10")

if final_score > 7:
    print("QUYẾT ĐỊNH: Chuyển cho CHUYÊN GIA.")
elif final_score > 4:
    print("QUYẾT ĐỊNH: Chuyển cho NHÂN VIÊN.")
else:
    print("QUYẾT ĐỊNH: AI BOT tự xử lý.")