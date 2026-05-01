import ollama
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import re

def ai_ranking(factor_name, user_description):
    prompt = f"Mô tả: '{user_description}'. Hãy chấm điểm mức độ của '{factor_name}' từ 1 đến 10 (1 là thấp nhất, 10 là cao nhất). Chỉ trả về duy nhất 1 con số nguyên."
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
    res_text = response['message']['content'].strip()
    nums = re.findall(r'\d+', res_text)
    return float(nums[0]) if nums else 5.0


demand = ctrl.Antecedent(np.arange(0, 11, 1), 'demand')
pressure = ctrl.Antecedent(np.arange(0, 11, 1), 'pressure')
reputation = ctrl.Antecedent(np.arange(0, 11, 1), 'reputation')
margin = ctrl.Antecedent(np.arange(0, 11, 1), 'margin')
seasonal = ctrl.Antecedent(np.arange(0, 11, 1), 'seasonal')
discount = ctrl.Consequent(np.arange(0, 71, 1), 'discount')

for var in [demand, pressure, reputation, margin, seasonal]:
    var.automf(3, names=['low', 'medium', 'high'])


discount['rat_thap'] = fuzz.trimf(discount.universe, [0, 0, 10])
discount['thap'] = fuzz.trimf(discount.universe, [5, 12, 25])
discount['trung_binh'] = fuzz.trimf(discount.universe, [20, 30, 45])
discount['cao'] = fuzz.trimf(discount.universe, [40, 50, 65])
discount['rat_cao'] = fuzz.trimf(discount.universe, [55, 70, 70])


rules = []
states = ['low', 'medium', 'high']

for d in states:
    for p in states:
        for r in states:
            for m in states:
                for s in states:
                    
                    score = states.index(d) + states.index(p) + states.index(r) + states.index(m) + states.index(s)
                    
                    if score <= 2:
                        out = 'rat_thap'
                    elif score <= 4:
                        out = 'thap'
                    elif score <= 6:
                        out = 'trung_binh'
                    elif score <= 8:
                        out = 'cao'
                    else:
                        out = 'rat_cao'
                    
                    
                    rule = ctrl.Rule(demand[d] & pressure[p] & reputation[r] & margin[m] & seasonal[s], discount[out])
                    rules.append(rule)


shopee_system = ctrl.ControlSystem(rules)
calc = ctrl.ControlSystemSimulation(shopee_system)

questions = {
    'demand': "nhu cầu khách hàng",
    'pressure': "áp lực đối thủ",
    'reputation': "uy tín cửa hàng",
    'margin': "biên lợi nhuận",
    'seasonal': "nhu cầu mùa vụ"
}

print(f"--- ĐÃ KHỞI TẠO HỆ THỐNG {len(rules)} LUẬT MỜ TỰ ĐỘNG ---")

for key, q in questions.items():
    desc = input(f"Miêu tả {q}: ")
    val = ai_ranking(key, desc)
    print(f"-> Llama3 chấm điểm: {val}/10")
    calc.input[key] = val


calc.compute()
print("\n" + "="*45)
print(f"PHẦN TRĂM GIẢM GIÁ TỐI ƯU: {round(calc.output['discount'], 2)}%")
print(f"Dựa trên phân tích toàn diện {len(rules)} kịch bản.")
print("="*45)