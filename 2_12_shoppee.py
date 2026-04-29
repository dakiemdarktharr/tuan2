import ollama
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import json

def ai_scoring_shopee(uy_tin, nhu_cau, loi_nhuan, mua_vu, doi_thu):
    prompt = f"""
    As a Shopee Strategy Expert, analyze and score (1-10):
    1. Store Rating: "{uy_tin}" (1: <4.0 stars, 10: >4.5 stars)
    2. Product Demand: "{nhu_cau}" (1: Low, 10: Very High)
    3. Profit Margin: "{loi_nhuan}" (1: Low, 10: High)
    4. Seasonal Event: "{mua_vu}" (1: None, 10: Mega Campaign 11.11/12.12)
    5. Competitor Discounts: "{doi_thu}" (1: Low, 10: Aggressive)

    Return ONLY JSON: {{"uy_tin": x, "nhu_cau": x, "loi_nhuan": x, "mua_vu": x, "doi_thu": x, "chiet_khau_goi_y": x}}
    """
    try:
        response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
        noi_dung = response['message']['content']
        du_lieu = json.loads(noi_dung[noi_dung.find('{'):noi_dung.rfind('}') + 1])
        return du_lieu
    except:
        return {"uy_tin": 5, "nhu_cau": 5, "loi_nhuan": 5, "mua_vu": 5, "doi_thu": 5, "chiet_khau_goi_y": 15}



uy_tin_cua_hang = ctrl.Antecedent(np.arange(0, 11, 1), 'uy_tin_cua_hang')
nhu_cau_san_pham = ctrl.Antecedent(np.arange(0, 11, 1), 'nhu_cau_san_pham')
bien_loi_nhuan = ctrl.Antecedent(np.arange(0, 11, 1), 'bien_loi_nhuan')
su_kien_mua_vu = ctrl.Antecedent(np.arange(0, 11, 1), 'su_kien_mua_vu')
chiet_khau_doi_thu = ctrl.Antecedent(np.arange(0, 11, 1), 'chiet_khau_doi_thu')



ty_le_chiet_khau = ctrl.Consequent(np.arange(0, 71, 1), 'ty_le_chiet_khau')


uy_tin_cua_hang.automf(3, names=['thap', 'trung_binh', 'cao'])
nhu_cau_san_pham.automf(3, names=['thap', 'trung_binh', 'cao'])
bien_loi_nhuan.automf(3, names=['thap', 'trung_binh', 'cao'])
su_kien_mua_vu.automf(3, names=['khong', 'vua', 'cao'])
chiet_khau_doi_thu.automf(3, names=['thap', 'trung_binh', 'cao'])

ty_le_chiet_khau['rat_thap'] = fuzz.trimf(ty_le_chiet_khau.universe, [0, 0, 5])
ty_le_chiet_khau['thap'] = fuzz.trimf(ty_le_chiet_khau.universe, [5, 10, 15])
ty_le_chiet_khau['trung_binh'] = fuzz.trimf(ty_le_chiet_khau.universe, [10, 20, 30])
ty_le_chiet_khau['cao'] = fuzz.trimf(ty_le_chiet_khau.universe, [20, 40, 50])
ty_le_chiet_khau['rat_cao'] = fuzz.trimf(ty_le_chiet_khau.universe, [40, 70, 70])




danh_sach_luat = []
for ut in ['thap', 'trung_binh', 'cao']:
    for nc in ['thap', 'trung_binh', 'cao']:
        for ln in ['thap', 'trung_binh', 'cao']:
            diem = 0
            if ln == 'thap': diem += 2
            if ut == 'cao': diem += 1
            if nc == 'cao': diem -= 1
            
            if diem >= 2: kq = ty_le_chiet_khau['rat_thap']
            elif diem == 1: kq = ty_le_chiet_khau['thap']
            else: kq = ty_le_chiet_khau['trung_binh']
            
            danh_sach_luat.append(ctrl.Rule(uy_tin_cua_hang[ut] & nhu_cau_san_pham[nc] & bien_loi_nhuan[ln], kq))

danh_sach_luat.append(ctrl.Rule(su_kien_mua_vu['cao'] & chiet_khau_doi_thu['cao'], ty_le_chiet_khau['rat_cao']))
danh_sach_luat.append(ctrl.Rule(su_kien_mua_vu['khong'] & chiet_khau_doi_thu['thap'] & bien_loi_nhuan['thap'], ty_le_chiet_khau['rat_thap']))

he_thong_shopee = ctrl.ControlSystem(danh_sach_luat)
mo_phong = ctrl.ControlSystemSimulation(he_thong_shopee)

def chay_chien_luoc_shopee():
    ut_in = input("Uy tín cửa hàng (Số sao/Đánh giá): ")
    nc_in = input("Nhu cầu sản phẩm (Thị hiếu): ")
    ln_in = input("Biên lợi nhuận (Chi phí): ")
    mv_in = input("Sự kiện mùa vụ (Ngày lễ/Flash Sale): ")
    dt_in = input("Chiết khấu đối thủ: ")

    kq = ai_scoring_shopee(ut_in, nc_in, ln_in, mv_in, dt_in)

    
    mo_phong.input['uy_tin_cua_hang'] = kq['uy_tin']
    mo_phong.input['nhu_cau_san_pham'] = kq['nhu_cau']
    mo_phong.input['bien_loi_nhuan'] = kq['loi_nhuan']
    mo_phong.input['su_kien_mua_vu'] = kq['mua_vu']
    mo_phong.input['chiet_khau_doi_thu'] = kq['doi_thu']
    mo_phong.compute()

    ket_qua_ck = mo_phong.output['ty_le_chiet_khau']

    
    print("="*50)
    print(f"ĐIỂM AI PHÂN TÍCH (1-10):")
    print(f"- Uy tín: {kq['uy_tin']} | Nhu cầu: {kq['nhu_cau']} | Lợi nhuận: {kq['loi_nhuan']}")
    print(f"- Mùa vụ: {kq['mua_vu']} | Đối thủ: {kq['doi_thu']}")
    print("-" * 50)
    print(f"CHIẾT KHẤU ĐỀ XUẤT TỪ HỆ THỐNG: {round(ket_qua_ck, 2)}%")
    print(f"CHIẾT KHẤU AI GỢI Ý: {kq['chiet_khau_goi_y']}%")
    print("="*50)

chay_chien_luoc_shopee()