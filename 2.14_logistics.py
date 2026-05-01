import ollama
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import re

def ai_rank(yeu_to, mo_ta):
    prompt = f"Ngữ cảnh Logistics: '{mo_ta}'. Chấm điểm '{yeu_to}' từ 1 đến 10. Chỉ trả về 1 con số nguyên."
    res = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
    nd = res['message']['content'].strip()
    so = re.findall(r'\d+', nd)
    return float(so[0]) if so else 5.0

density = ctrl.Antecedent(np.arange(0, 11, 1), 'density')
urgency = ctrl.Antecedent(np.arange(0, 11, 1), 'urgency')
load = ctrl.Antecedent(np.arange(0, 11, 1), 'load')
traffic = ctrl.Antecedent(np.arange(0, 11, 1), 'traffic')
profit = ctrl.Antecedent(np.arange(0, 11, 1), 'profit')

combine = ctrl.Consequent(np.arange(0, 11, 1), 'combine')
priority = ctrl.Consequent(np.arange(0, 11, 1), 'priority')

for var in [density, urgency, load, traffic, profit]:
    var.automf(3, names=['low', 'medium', 'high'])

combine.automf(3, names=['it', 'mot_so', 'nhieu'])
priority.automf(3, names=['thap', 'trung_binh', 'cao'])

rules_combine = []
rules_priority = []
st = ['low', 'medium', 'high']

for d in st:
    for u in st:
        for l in st:
            for t in st:
                for p in st:
                    score_c = st.index(d) - st.index(l) - st.index(t)
                    if score_c <= -1: oc = 'it'
                    elif score_c <= 1: oc = 'mot_so'
                    else: oc = 'nhieu'
                    rules_combine.append(ctrl.Rule(density[d] & urgency[u] & load[l] & traffic[t] & profit[p], combine[oc]))
                    
                    score_p = st.index(u) + st.index(p) - st.index(t)
                    if score_p <= -1: op = 'thap'
                    elif score_p <= 1: op = 'trung_binh'
                    else: op = 'cao'
                    rules_priority.append(ctrl.Rule(density[d] & urgency[u] & load[l] & traffic[t] & profit[p], priority[op]))

logistics_ctrl = ctrl.ControlSystem(rules_combine + rules_priority)
simulation = ctrl.ControlSystemSimulation(logistics_ctrl)

cau_hoi = {
    'density': 'mật độ đơn hàng tại khu vực',
    'urgency': 'mức độ khẩn cấp của các đơn',
    'load': 'tải trọng hiện tại của tài xế',
    'traffic': 'tình trạng giao thông hiện tại',
    'profit': 'lợi nhuận trên mỗi chuyến giao'
}


for i, j in cau_hoi.items():
    cautrl = input(f"Mô tả {j}: ")
    diem = ai_rank(i, cautrl)
    simulation.input[i] = diem

simulation.compute()

print("="*40)
print(f"số lượng đơn kết hợp: {round(simulation.output['combine'], 1)} đơn/chuyến")
print(f"ưu tiên giao hàng: {round(simulation.output['priority'], 1)}/10")
print("="*40)