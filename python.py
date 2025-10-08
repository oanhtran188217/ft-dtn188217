# python.py

import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError
import os
import time # Thư viện hữu ích cho việc mô phỏng độ trễ

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # ******************************* PHẦN SỬA LỖI BẮT ĐẦU *******************************
    # Lỗi xảy ra khi dùng .replace() trên giá trị đơn lẻ (numpy.int64).
    # Sử dụng điều kiện ternary để xử lý giá trị 0 thủ công cho mẫu số.
    
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    # ******************************* PHẦN SỬA LỖI KẾT THÚC *******************************
    
    return df

# --- Hàm gọi API Gemini cho Nhận xét (Giữ nguyên) ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra cấu hình Secrets trên Streamlit Cloud."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- Hàm chuyển đổi định dạng tin nhắn cho Gemini Chat (MỚI) ---
def to_gemini_format(st_messages):
    """Chuyển đổi lịch sử chat của Streamlit sang định dạng API Gemini."""
    gemini_messages = []
    for message in st_messages:
        # Streamlit 'role' ('user', 'assistant') -> Gemini 'role' ('user', 'model')
        role = "user" if message["role"] == "user" else "model"
        
        # Chỉ lấy tin nhắn văn bản
        gemini_messages.append({
            "role": role,
            "parts": [{"text": message["content"]}]
        })
    return gemini_messages

# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

# Khởi tạo khóa API, ưu tiên lấy từ st.secrets (dùng cho Chatbot và Nhận xét AI)
API_KEY = st.secrets.get("GEMINI_API_KEY")

if uploaded_file is not None:
    # ... (Các bước xử lý và hiển thị dữ liệu như cũ) ...
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            try:
                # Lọc giá trị cho Chỉ số Thanh toán Hiện hành (Ví dụ)
                
                # Lấy Tài sản ngắn hạn
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Lấy Nợ ngắn hạn (Dùng giá trị giả định hoặc lọc từ file nếu có)
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tính toán, xử lý chia cho 0
                thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N if no_ngan_han_N != 0 else float('inf')
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1 if no_ngan_han_N_1 != 0 else float('inf')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=f"{thanh_toan_hien_hanh_N_1:.2f} lần" if thanh_toan_hien_hanh_N_1 != float('inf') else "∞"
                    )
                with col2:
                    delta_value = thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1 if thanh_toan_hien_hanh_N != float('inf') and thanh_toan_hien_hanh_N_1 != float('inf') else "N/A"
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần" if thanh_toan_hien_hanh_N != float('inf') else "∞",
                        delta=f"{delta_value:.2f}" if isinstance(delta_value, float) else None
                    )
                    
            except IndexError:
                 st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
                 thanh_toan_hien_hanh_N = "N/A" # Dùng để tránh lỗi ở Chức năng 5
                 thanh_toan_hien_hanh_N_1 = "N/A"
            except ZeroDivisionError:
                 st.warning("Lỗi: Nợ Ngắn Hạn bằng 0. Chỉ số thanh toán hiện hành rất lớn (vô cùng).")
                 thanh_toan_hien_hanh_N = "Inf"
                 thanh_toan_hien_hanh_N_1 = "Inf"
                
            # --- Chức năng 5: Nhận xét AI ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
            
            # Chuẩn bị dữ liệu để gửi cho AI
            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Tăng trưởng Tài sản ngắn hạn (%)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%" if not df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)].empty else "N/A", 
                    f"{thanh_toan_hien_hanh_N_1}", 
                    f"{thanh_toan_hien_hanh_N}"
                ]
            }).to_markdown(index=False) 

            if st.button("Yêu cầu AI Phân tích"):
                if API_KEY:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai, API_KEY)
                    st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                    st.info(ai_result)
                else:
                    st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")

# -----------------------------------------------------------
# --------------- PHẦN TÍCH HỢP CHATBOT GEMINI MỚI ---------------
# -----------------------------------------------------------

st.divider()

# Khởi tạo lịch sử chat trong session_state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Chào bạn! Tôi là Gemini. Bạn có thể hỏi tôi thêm về dữ liệu đã tải hoặc các khái niệm tài chính khác."})

st.subheader("6. Chat Hỏi đáp với Gemini 💬")

# 1. Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Xử lý khung nhập liệu
if prompt := st.chat_input("Hỏi Gemini về báo cáo tài chính hoặc bất cứ điều gì..."):
    
    if not API_KEY:
        st.error("Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Không thể chat.")
    else:
        # Thêm tin nhắn người dùng vào lịch sử và hiển thị
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Chuẩn bị ngữ cảnh (context)
        # Nếu có dữ liệu đã xử lý, thêm nó vào ngữ cảnh để Gemini trả lời tốt hơn
        context_data = ""
        if 'df_processed' in locals() and df_processed is not None:
             context_data = "\n\nDữ liệu Báo cáo Tài chính đã xử lý (Tốc độ tăng trưởng và Tỷ trọng): \n" + df_processed.to_markdown(index=False)
        
        # 3. Bắt đầu trả lời
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                client = genai.Client(api_key=API_KEY)
                model_name = "gemini-2.5-flash"
                
                # Lấy lịch sử tin nhắn Streamlit (trừ tin nhắn cuối cùng vừa thêm)
                st_history = st.session_state.messages[:-1]
                
                # Chuyển đổi lịch sử sang định dạng Gemini
                gemini_history = to_gemini_format(st_history)

                # Bổ sung dữ liệu phân tích vào tin nhắn đầu tiên của phiên chat
                # để cung cấp ngữ cảnh chuyên biệt khi cần.
                if context_data and not any("Dữ liệu Báo cáo Tài chính" in m["parts"][0]["text"] for m in gemini_history if m["role"] == "user"):
                    # Thêm dữ liệu vào tin nhắn PROMPT
                    full_prompt = prompt + context_data
                else:
                    full_prompt = prompt

                # Gọi API Gemini với chế độ Streaming
                response_stream = client.models.generate_content_stream(
                    model=model_name,
                    contents=gemini_history + [{"role": "user", "parts": [{"text": full_prompt}]}]
                )

                # Hiển thị phản hồi streaming
                for chunk in response_stream:
                    if chunk.text:
                        full_response += chunk.text
                        message_placeholder.markdown(full_response + "▌")
                        
                message_placeholder.markdown(full_response) # Hiển thị phản hồi cuối cùng
                
            except APIError as e:
                error_message = f"Lỗi API Gemini: {e}"
                message_placeholder.markdown(error_message)
                full_response = error_message
            except Exception as e:
                error_message = f"Đã xảy ra lỗi không xác định: {e}"
                message_placeholder.markdown(error_message)
                full_response = error_message

        # 4. Thêm phản hồi của trợ lý vào lịch sử chat của Streamlit
        st.session_state.messages.append({"role": "assistant", "content": full_response})
