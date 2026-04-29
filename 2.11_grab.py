import ollama
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import json

def ai_scoring_engine(w_text, d_text, p_text):
    reference_formula = """
    Fare_Reference = (Base_12500 + (Dist_Estimate * 4300)) * Multiplier + Platform_2000
    Total = Fare_Reference * 1.1
    """
    
    prompt = f"""
    As a Grab pricing expert, analyze these inputs and provide intensity scores (1-10):
    1. Weather description: "{w_text}" -> Score 1: Clear, 10: Storm/Extreme.
    2. Demand description: "{d_text}" -> Score 1: Empty, 10: Overloaded.
    3. Distance: "{p_text}" -> Score 1: 0-1km, 10: 20-30km.

    Reference Formula for calculation context: {reference_formula}

    Return ONLY a JSON object with these keys: "weather", "demand", "peak", "total_fare"
    Example: {{"weather": 8, "demand": 9, "distance": 10, "total_fare": 85000}}
    """
    
    try:
        response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
        content = response['message']['content']
        data = json.loads(content[content.find('{'):content.rfind('}') + 1])
        return data
    except:
        return {"weather": 5, "demand": 5, "distance": 5, "total_fare": 45000}

weather = ctrl.Antecedent(np.arange(0, 11, 1), 'weather')
demand = ctrl.Antecedent(np.arange(0, 11, 1), 'demand')
distance = ctrl.Antecedent(np.arange(0, 11, 1), 'distance')
multiplier = ctrl.Consequent(np.arange(1, 3.1, 0.1), 'multiplier')

weather.automf(3, names=['low', 'medium', 'high'])
demand.automf(3, names=['low', 'medium', 'high'])
distance.automf(3, names=['low', 'medium', 'high'])

multiplier['low'] = fuzz.trimf(multiplier.universe, [1, 1, 1.5])
multiplier['medium'] = fuzz.trimf(multiplier.universe, [1.2, 1.8, 2.5])
multiplier['high'] = fuzz.trimf(multiplier.universe, [2, 3, 3])

rules = []
states = ['low', 'medium', 'high']

for w in states:
    for d in states:
        for p in states:
            score = 0
            if w == 'high': score += 2
            if d == 'high': score += 2
            if p == 'high': score += 2
            if w == 'medium': score += 1
            if d == 'medium': score += 1
            if p == 'medium': score += 1
            
            if score >= 4:
                res = multiplier['high']
            elif score >= 2:
                res = multiplier['medium']
            else:
                res = multiplier['low']
            
            rules.append(ctrl.Rule(weather[w] & demand[d] & distance[p], res))

pricing_ctrl = ctrl.ControlSystem(rules)
pricing_sim = ctrl.ControlSystemSimulation(pricing_ctrl)

def thuc_thi_he_thong_diem():
    
    w_input = input("Mô tả thời tiết (bằng chữ): ")
    d_input = input("Mô tả nhu cầu khách (bằng chữ): ")
    p_input = input("Mô tả độ dài đoạn đường (bằng chữ): ")
    
    ai_res = ai_scoring_engine(w_input, d_input, p_input)
    
    pricing_sim.input['weather'] = ai_res['weather']
    pricing_sim.input['demand'] = ai_res['demand']
    pricing_sim.input['distance'] = ai_res['distance']
    pricing_sim.compute()
    
    he_so_mo = pricing_sim.output['multiplier']
    
    print("="*50)
    print("KẾT QUẢ PHÂN TÍCH TỪ AI:")
    print(f"- Điểm Thời tiết: {ai_res['weather']}/10")
    print(f"- Điểm Nhu cầu: {ai_res['demand']}/10")
    print(f"- Đoạn đường: {ai_res['distance']}/10")
    print("-" * 50)
    print(f"Hệ số nhân từ Logic mờ: x{round(he_so_mo, 2)}")
    print(f"Giá tiền AI đề xuất: {round(ai_res['total_fare'], -2)} VNĐ")
    print("="*50)

thuc_thi_he_thong_diem()