import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os

quan_phuong_map = {
    "Bình Chánh": ["Thị trấn Tân Túc", "Xã An Phú Tây", "Xã Bình Chánh", "Xã Bình Hưng", "Xã Bình Lợi", "Xã Hưng Long", "Xã Lê Minh Xuân", "Xã Phong Phú", "Xã Phạm Văn Hai", "Xã Qui Đức", "Xã Tân Kiên", "Xã Tân Nhựt", "Xã Tân Quý Tây", "Xã Vĩnh Lộc A", "Xã Vĩnh Lộc B", "Xã Đa Phước"],
    "Bình Thạnh": ["Phường 1", "Phường 11", "Phường 12", "Phường 13", "Phường 14", "Phường 15", "Phường 17", "Phường 19", "Phường 2", "Phường 21", "Phường 22", "Phường 24", "Phường 25", "Phường 26", "Phường 27", "Phường 28", "Phường 3", "Phường 5", "Phường 6", "Phường 7"],
    "Bình Tân": ["Phường An Lạc", "Phường An Lạc A", "Phường Bình Hưng Hòa", "Phường Bình Hưng Hòa A", "Phường Bình Hưng Hòa B", "Phường Bình Trị Đông", "Phường Bình Trị Đông A", "Phường Bình Trị Đông B", "Phường Tân Tạo", "Phường Tân Tạo A"],
    "Cần Giờ": ["Thị trấn Cần Thạnh", "Xã Bình Khánh", "Xã Long Hòa", "Xã Lý Nhơn"],
    "Củ Chi": ["Thị trấn Củ Chi", "Xã An Nhơn Tây", "Xã Bình Mỹ", "Xã Hòa Phú", "Xã Nhuận Đức", "Xã Phú Hòa Đông", "Xã Phú Mỹ Hưng", "Xã Phước Hiệp", "Xã Phước Thạnh", "Xã Phước Vĩnh An", "Xã Thái Mỹ", "Xã Trung An", "Xã Trung Lập Hạ", "Xã Trung Lập Thượng", "Xã Tân An Hội", "Xã Tân Thông Hội", "Xã Tân Thạnh Tây", "Xã Tân Thạnh Đông"],
    "Gò Vấp": ["Phường 1", "Phường 10", "Phường 11", "Phường 12", "Phường 13", "Phường 14", "Phường 15", "Phường 16", "Phường 17", "Phường 3", "Phường 4", "Phường 5", "Phường 6", "Phường 7", "Phường 8", "Phường 9"],
    "Hóc Môn": ["Thị trấn Hóc Môn", "Xã Bà Điểm", "Xã Thới Tam Thôn", "Xã Trung Chánh", "Xã Tân Hiệp", "Xã Tân Thới Nhì", "Xã Tân Xuân", "Xã Xuân Thới Sơn", "Xã Xuân Thới Thượng", "Xã Xuân Thới Đông", "Xã Đông Thạnh"],
    "Nhà Bè": ["Thị trấn Nhà Bè", "Xã Hiệp Phước", "Xã Long Thới", "Xã Nhơn Đức", "Xã Phú Xuân", "Xã Phước Kiển", "Xã Phước Lộc"],
    "Phú Nhuận": ["Phường 1", "Phường 10", "Phường 11", "Phường 12", "Phường 13", "Phường 14", "Phường 15", "Phường 17", "Phường 2", "Phường 3", "Phường 4", "Phường 5", "Phường 7", "Phường 8", "Phường 9"],
    "Quận 1": ["Phường 1", "Phường 10", "Phường 11", "Phường 12", "Phường 13", "Phường 14", "Phường 15", "Phường 16", "Phường 2", "Phường 3", "Phường 4", "Phường 5", "Phường 6", "Phường 7", "Phường 8", "Phường 9", "Phường An Phú Đông", "Phường Bến Nghé", "Phường Bến Thành", "Phường Cô Giang", "Phường Cầu Kho", "Phường Cầu Ông Lãnh", "Phường Hiệp Thành", "Phường Nguyễn Cư Trinh", "Phường Nguyễn Thái Bình", "Phường Phạm Ngũ Lão", "Phường Thạnh Lộc", "Phường Thạnh Xuân", "Phường Thới An", "Phường Trung Mỹ Tây", "Phường Tân Chánh Hiệp", "Phường Tân Hưng Thuận", "Phường Tân Thới Hiệp", "Phường Tân Thới Nhất", "Phường Tân Định", "Phường Đa Kao", "Phường Đông Hưng Thuận", "phường Cô Giang"],
    "Quận 3": ["Phường 1", "Phường 10", "Phường 11", "Phường 12", "Phường 13", "Phường 14", "Phường 2", "Phường 3", "Phường 4", "Phường 5", "Phường 9", "Phường Võ Thị Sáu"],
    "Quận 4": ["Phường 1", "Phường 10", "Phường 12", "Phường 13", "Phường 14", "Phường 15", "Phường 16", "Phường 18", "Phường 2", "Phường 3", "Phường 4", "Phường 5", "Phường 6", "Phường 8", "Phường 9", "Phường Khánh Hội"],
    "Quận 5": ["Phường 1", "Phường 10", "Phường 11", "Phường 12", "Phường 13", "Phường 14", "Phường 15", "Phường 2", "Phường 3", "Phường 4", "Phường 5", "Phường 6", "Phường 7", "Phường 8", "Phường 9", "Phường Chợ lớn tphcm"],
    "Quận 6": ["Phường 1", "Phường 10", "Phường 11", "Phường 12", "Phường 13", "Phường 14", "Phường 2", "Phường 3", "Phường 4", "Phường 5", "Phường 6", "Phường 7", "Phường 8", "Phường 9"],
    "Quận 7": ["Phường Bình Thuận", "Phường Phú Mỹ", "Phường Phú Thuận", "Phường Tân Hưng", "Phường Tân Kiểng", "Phường Tân Phong", "Phường Tân Phú", "Phường Tân Quy", "Phường Tân Thuận Tây", "Phường Tân Thuận Đông"],
    "Quận 8": ["Phường 1", "Phường 10", "Phường 12", "Phường 13", "Phường 14", "Phường 15", "Phường 16", "Phường 2", "Phường 3", "Phường 4", "Phường 5", "Phường 6", "Phường 7", "Phường 8", "Phường 9", "Phường Rạch Ông"],
    "Thủ Đức": ["Phường  Thạnh Mỹ Lợi", "Phường An Khánh", "Phường An Lợi Đông", "Phường An Phú", "Phường Bình An", "Phường Bình Chiểu", "Phường Bình Khánh", "Phường Bình Thọ", "Phường Bình Trưng Tây", "Phường Bình Trưng Đông", "Phường Cát Lái", "Phường Hiệp Bình Chánh", "Phường Hiệp Bình Phước", "Phường Hiệp Phú", "Phường Linh Chiểu", "Phường Linh Trung", "Phường Linh Tây", "Phường Linh Xuân", "Phường Linh Đông", "Phường Long Bình", "Phường Long Phước", "Phường Long Thạnh Mỹ", "Phường Long Trường", "Phường Phú Hữu", "Phường Phước Bình", "Phường Phước Long A", "Phường Phước Long B", "Phường Tam Bình", "Phường Tam Phú", "Phường Thạnh Mỹ Lợi", "Phường Thảo Điền", "Phường Thủ Thiêm", "Phường Trường Thạnh", "Phường Trường Thọ", "Phường Tân Phú", "Phường Tăng Nhơn Phú A", "Phường Tăng Nhơn Phú B"],
    "Tân Bình": ["Phường 1", "Phường 10", "Phường 11", "Phường 12", "Phường 13", "Phường 14", "Phường 15", "Phường 2", "Phường 3", "Phường 4", "Phường 5", "Phường 6", "Phường 7", "Phường 8", "Phường 9"],
    "Tân Phú": ["Phường Hiệp Tân", "Phường Hòa Thạnh", "Phường Phú Thạnh", "Phường Phú Thọ Hòa", "Phường Phú Trung", "Phường Sơn Kỳ", "Phường Tân Phú", "Phường Tân Quý", "Phường Tân Sơn Nhì", "Phường Tân Thành", "Phường Tân Thới Hòa", "Phường Tây Thạnh", "Xã Tân Phú Trung"]
}

phap_ly_list = ['Sổ riêng', 'Sổ chung', 'Hợp đồng mua bán', 'Đang chờ sổ', 'Vi bằng / uỷ quyền', 'Không rõ']
noi_that_list = ['Nội thất cơ bản', 'Full nội thất', 'Nội thất cao cấp', 'Không nội thất', 'Không rõ']

def cap_nhat_phuong(quan_duoc_chon):
    danh_sach_phuong = quan_phuong_map.get(quan_duoc_chon, [])
    return gr.update(choices=danh_sach_phuong, value=danh_sach_phuong[0] if danh_sach_phuong else None)

def feature_engineering(data):
    df_fe = data.copy()
    
    # 1. Phân khúc diện tích
    bins = [0, 30, 120, np.inf]
    labels = ['Duoi_30', 'Tu_30_den_120', 'Tren_120']
    df_fe['Phan_khuc_dien_tich'] = pd.cut(df_fe['Diện tích'], bins=bins, labels=labels)
    
    # 2. Tỷ lệ WC/Phòng ngủ
    df_fe['Ty_le_Bath_Bed'] = df_fe['Số phòng tắm, vệ sinh'] / (df_fe['Số phòng ngủ'] + 1e-5)
    
    # 3. CÁC BIẾN TƯƠNG TÁC KẾT CẤU & VỊ TRÍ
    df_fe['Mat_tien_Quan'] = df_fe['Mặt tiền'].astype(int).astype(str) + '_' + df_fe['Quận'].astype(str)
    df_fe['Cao_tang_Quan'] = df_fe['Cao tầng'].astype(int).astype(str) + '_' + df_fe['Quận'].astype(str)
    df_fe['Mat_tien_Cao_tang'] = df_fe['Mặt tiền'].astype(int).astype(str) + '_' + df_fe['Cao tầng'].astype(int).astype(str)
    
    # 4. TIỆN ÍCH (Khu biệt lập & Tổng tiện ích)
    df_fe['Khu_biet_lap'] = np.where(
        (df_fe['Gần chợ'] == 0) & (df_fe['Gần trường học'] == 0) & (df_fe['Gần bệnh viện'] == 0), 1, 0
    )
    df_fe['Tong_tien_ich'] = df_fe['Gần bệnh viện'] + df_fe['Gần chợ'] + df_fe['Gần trường học']
    
    # 5. XỬ LÝ DỮ LIỆU ĐỊNH DANH (Category)
    if 'Cluster' in df_fe.columns:
        df_fe['Cluster'] = df_fe['Cluster'].astype(str)
        
    # Xóa các biến tiện ích lẻ 
    util_cols = ['Gần bệnh viện', 'Gần chợ', 'Gần trường học']
    df_fe = df_fe.drop(columns=[col for col in util_cols if col in df_fe.columns])
    
    return df_fe

def predict_price(model_name, dien_tich_str, so_phong_ngu, so_phong_tam, phap_ly, noi_that, 
                  mat_tien, gan_bv, gan_cho, gan_th, cao_tang, quy_hoach, phuong, quan):
    
    try:
        dt_val = str(dien_tich_str).replace(',', '.')
        dien_tich = float(dt_val)
        if dien_tich <= 0: return "Lỗi", "Diện tích phải > 0"
    except:
        return "Lỗi định dạng", "Vui lòng nhập diện tích hợp lệ (VD: 26.9)"

    cluster_pipeline_path = "models/classification/xgb_pipeline.pkl"
    feature_columns_path = "models/classification/feature_columns.pkl"
    
    if not os.path.exists(cluster_pipeline_path) or not os.path.exists(feature_columns_path):
        return "Lỗi", "Không tìm thấy file mô hình phân cụm (xgb_pipeline.pkl / feature_columns.pkl)"
    
    try:
        cluster_pipeline = joblib.load(cluster_pipeline_path)
        feature_cols = joblib.load(feature_columns_path)
        
        # Prepare input data cho Pipeline phân cụm (Loại bỏ Diện tích theo yêu cầu)
        cluster_input = {
            'Số phòng ngủ': so_phong_ngu,
            'Số phòng tắm, vệ sinh': so_phong_tam,
            'Pháp lý': phap_ly,
            'Nội thất': noi_that,
            'Mặt tiền': 1 if mat_tien else 0,
            'Gần bệnh viện': 1 if gan_bv else 0,
            'Gần chợ': 1 if gan_cho else 0,
            'Gần trường học': 1 if gan_th else 0,
            'Cao tầng': 1 if cao_tang else 0,
            'Quy hoạch': 1 if quy_hoach else 0,
            'Phường': phuong,
            'Quận': quan
        }
        
        df_cluster = pd.DataFrame([cluster_input])
        df_cluster = df_cluster[feature_cols] # Ensure column order matches training
        
        # Dự đoán cụm
        assigned_cluster = str(int(cluster_pipeline.predict(df_cluster)[0]))
        
    except Exception as e:
        return "Lỗi phân cụm", str(e)

    numeric_features = {
        'Diện tích': dien_tich,
        'Số phòng ngủ': so_phong_ngu,
        'Số phòng tắm, vệ sinh': so_phong_tam,
        'Mặt tiền': 1 if mat_tien else 0,
        'Gần bệnh viện': 1 if gan_bv else 0,
        'Gần chợ': 1 if gan_cho else 0,
        'Gần trường học': 1 if gan_th else 0,
        'Cao tầng': 1 if cao_tang else 0,
        'Quy hoạch': 1 if quy_hoach else 0
    }
    
    input_df = pd.DataFrame([{
        **numeric_features,
        'Pháp lý': phap_ly,
        'Nội thất': noi_that,
        'Phường': phuong,
        'Quận': quan,
        'Cluster': assigned_cluster
    }])
    
    model_path = f"models/tuned_model/best_{'xgb' if model_name == 'XGBoost' else 'rf'}_model.pkl"
    if not os.path.exists(model_path):
        model_path = f"best_{'xgb' if model_name == 'XGBoost' else 'rf'}_model.pkl"
        if not os.path.exists(model_path):
            return "Lỗi", f"Không tìm thấy file {model_path}"

    try:
        model = joblib.load(model_path)
        data_fe = feature_engineering(input_df)
        
        pred_log = model.predict(data_fe)[0]
        pred_val = np.expm1(pred_log) # Tỷ VNĐ/m2
        
        return f"{pred_val * 1000:,.1f} Triệu VNĐ/m²", f"{pred_val * dien_tich:,.3f} Tỷ VNĐ (Cụm: {assigned_cluster})"
    except Exception as e:
        return "Lỗi hệ thống", str(e)

with gr.Blocks(title="Dự đoán Giá Nhà TP.HCM") as demo:
    gr.Markdown("# Hệ Thống Dự Đoán Giá Nhà (TP.HCM)")
    gr.Markdown("Xác định giá nhà dựa trên vị trí, diện tích và các tiện ích đi kèm.")
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Thông tin nhà")
            with gr.Row():
                quan = gr.Dropdown(choices=list(quan_phuong_map.keys()), label="Quận", value="Quận 1")
                phuong = gr.Dropdown(choices=quan_phuong_map["Quận 1"], label="Phường", value="Phường Bến Nghé")
            
            dien_tich = gr.Textbox(label="Diện tích (m²)", value="50.0", placeholder="VD: 26.9")
            
            with gr.Row():
                so_phong_ngu = gr.Number(label="Số phòng ngủ", value=2, minimum=0)
                so_phong_tam = gr.Number(label="Số phòng tắm/WC", value=2, minimum=0)
                
            with gr.Row():
                phap_ly = gr.Dropdown(choices=phap_ly_list, label="Pháp lý", value="Sổ riêng")
                noi_that = gr.Dropdown(choices=noi_that_list, label="Nội thất", value="Nội thất cơ bản")
            
        with gr.Column(scale=1):
            gr.Markdown("### Tiện ích & Đặc điểm")
            mat_tien = gr.Checkbox(label="Mặt tiền (Mặt phố)")
            gan_bv = gr.Checkbox(label="Gần bệnh viện")
            gan_cho = gr.Checkbox(label="Gần chợ")
            gan_th = gr.Checkbox(label="Gần trường học")
            cao_tang = gr.Checkbox(label="Nhà cao tầng")
            quy_hoach = gr.Checkbox(label="Nằm trong quy hoạch")
            
    quan.change(fn=cap_nhat_phuong, inputs=quan, outputs=phuong)

    gr.Markdown("---")
    model_name = gr.Radio(choices=["XGBoost", "Random Forest"], label="Chọn Mô Hình Dự Đoán Giá", value="XGBoost")
    btn = gr.Button("DỰ ĐOÁN GIÁ", variant="primary", size="lg")
    
    with gr.Row():
        out_gia_m2 = gr.Textbox(label="Giá dự kiến trên 1m²", text_align="center")
        out_tong_gia = gr.Textbox(label="Tổng giá trị dự kiến & Phân cụm", text_align="center")

    btn.click(
        fn=predict_price,
        inputs=[model_name, dien_tich, so_phong_ngu, so_phong_tam, phap_ly, noi_that, 
                mat_tien, gan_bv, gan_cho, gan_th, cao_tang, quy_hoach, phuong, quan],
        outputs=[out_gia_m2, out_tong_gia]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), ssr_mode=False)