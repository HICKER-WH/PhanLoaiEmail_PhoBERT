from typing import Dict
class EmailAnalysisSystem:
    def __init__(self, spam_model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize PhoBERT classifier
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.spam_classifier = EmailClassifier(2)  # 2 classes: spam/not spam
        if spam_model_path:
            self.spam_classifier.load_state_dict(torch.load(spam_model_path))
        self.spam_classifier = self.spam_classifier.to(self.device)

        # Initialize importance evaluator
        self.importance_evaluator = ImportanceEvaluator()

    def analyze_email(self, email_content: str) -> Dict:
        # First, check if it's spam
        is_spam = self._check_spam(email_content)

        if is_spam:
            return {
                "is_spam": True,
                "importance_analysis": None,
                "recommendation": "This email has been classified as spam and should be ignored."
            }

        # If not spam, evaluate importance
        importance_analysis = self.importance_evaluator.evaluate_importance(email_content)

        return {
            "is_spam": False,
            "importance_analysis": importance_analysis,
            "recommendation": self._generate_recommendation(importance_analysis)
        }

    def _check_spam(self, email_content: str) -> bool:
        self.spam_classifier.eval()
        encoding = self.tokenizer.encode_plus(
            email_content,
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            outputs = self.spam_classifier(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

        return bool(preds.item())

    def _generate_recommendation(self, importance_analysis: Dict) -> str:
        # Generate recommendation based on importance analysis
        priority = importance_analysis.get('priority_level', 'Medium')
        urgency = importance_analysis.get('urgency', 3)
        required_action = importance_analysis.get('required_action', 'Unknown')

        if priority == 'High' and urgency >= 4:
            return "Immediate attention required. Handle this email as soon as possible."
        elif priority == 'Medium' or (urgency >= 3):
            return "Handle this email within normal business hours."
        else:
            return "Low priority. Handle when convenient."
from abacusai import ApiClient
import json

class ImportanceEvaluator:
    def __init__(self, api_key):
        self.client = ApiClient(api_key=api_key)
        # Khởi tạo chat session với GPT-4
        self.chat_session = self.client.create_chat_session(
            project_id="ad9544c9e",  # Sử dụng GPT-4
            # temperature=0.3,    # Độ sáng tạo thấp để có kết quả nhất quán
            # system_prompt="""Bạn là một trợ lý AI chuyên đánh giá tầm quan trọng của email.
            # Nhiệm vụ của bạn là phân tích email và đưa ra đánh giá chi tiết về mức độ khẩn cấp,
            # tác động đến công việc, yêu cầu hành động và mức độ ưu tiên."""
        )

    def evaluate_importance(self, email_content):
        try:
            # Xây dựng prompt
            prompt = f"""Hãy phân tích email sau có nội dung tố giác, khiếu nại không:

Email cần đánh giá:
---
{email_content}
---

Yêu cầu: Kiểm tra email có nội dung tố giác hay khiếu nại không, nếu không phải trả về kết quả là chữ 'scam' nếu có Phân tích và trả về kết quả theo 2 nội dung sau:
- Độ ưu tiên: Mức độ ưu tiên (Thấp/Trung bình/Cao)
- phân tích: phân tích ngắn gọn về nội dung

"""

            # Gọi Chat API với GPT-4
            # response = self.client.get_chat_response(
            #     chat_session_id=self.chat_session.chat_session_id,
            #     prompt=prompt
            # )
            response = self.client.evaluate_prompt(prompt = prompt, system_message = "Bạn phải trả lời một cách chính xác.", llm_name = "OPENAI_GPT4O")
            print(response.content)
            return response.content

        #     # Parse kết quả
        #     if hasattr(response, 'response'):
        #         try:
        #             # Tìm và trích xuất phần JSON từ response
        #             json_str = self._extract_json(response.response)
        #             result = json.loads(json_str)
        #             return self._validate_result(result)
        #         except json.JSONDecodeError as e:
        #             print(f"Lỗi parse JSON: {e}")
        #             return self._default_importance()
        #     else:
        #         return self._default_importance()

        except Exception as e:
            print(f"Lỗi khi gọi Chat API: {e}")
            return "Lỗi"


    def _extract_json(self, text):
        """Trích xuất phần JSON từ response text"""
        try:
            # Tìm dấu { đầu tiên và } cuối cùng
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end != 0:
                return text[start:end]
            return "{}"
        except:
            return "{}"

    def _validate_result(self, result):
        """Kiểm tra và chuẩn hóa kết quả"""
        validated = {}

        # Kiểm tra và chuẩn hóa urgency
        urgency = result.get('urgency', 3)
        validated['urgency'] = max(1, min(5, int(urgency)))

        # Kiểm tra và chuẩn hóa business_impact
        impact = result.get('business_impact', 3)
        validated['business_impact'] = max(1, min(5, int(impact)))

        # Kiểm tra và chuẩn hóa required_action
        action = result.get('required_action', 'Không xác định')
        validated['required_action'] = 'Có' if str(action).lower() in ['có', 'yes', 'true'] else 'Không'

        # Kiểm tra và chuẩn hóa priority_level
        priority = str(result.get('priority_level', 'Trung bình')).lower()
        if priority in ['cao', 'high']:
            validated['priority_level'] = 'Cao'
        elif priority in ['thấp', 'low']:
            validated['priority_level'] = 'Thấp'
        else:
            validated['priority_level'] = 'Trung bình'

        # Thêm phần analysis nếu có
        validated['analysis'] = result.get('analysis', 'Không có phân tích')

        return validated

    def _default_importance(self):
        """Trả về giá trị mặc định khi có lỗi"""
        return {
            'urgency': 3,
            'business_impact': 3,
            'required_action': 'Không xác định',
            'priority_level': 'Trung bình',
            'analysis': 'Không thể phân tích do lỗi hệ thống'
        }

import streamlit as st
import pandas as pd

import time

# Khởi tạo ImportanceEvaluator
@st.cache_resource
def get_evaluator():
    abacus_api_key = "s2_04004a405eb64d2490339f1ecbe9866a"
    return ImportanceEvaluator(abacus_api_key)

def main():
    st.set_page_config(page_title="Email Analysis", layout="wide")

    # Tiêu đề ứng dụng
    st.title("🔍 Hệ thống Phân tích Email")
    st.markdown("---")

    # Sidebar cho cấu hình
    with st.sidebar:
        st.header("⚙️ Cấu hình")
        batch_size = st.number_input("Số email phân tích mỗi lần", 
                                   min_value=1, 
                                   max_value=100, 
                                   value=10)

    # Upload file
    uploaded_file = st.file_uploader("📤 Upload file CSV chứa danh sách email", 
                                   type=['csv'])

    if uploaded_file is not None:
        try:
            # Đọc file CSV
            df = pd.read_csv(uploaded_file)

            # Hiển thị preview data
            st.subheader("📋 Preview dữ liệu")
            st.dataframe(df.head())

            # Chọn cột chứa nội dung email
            email_column = st.selectbox("Chọn cột chứa nội dung email", 
                                      df.columns)

            if st.button("🚀 Bắt đầu phân tích"):
                evaluator = get_evaluator()
                emails = df[email_column].tolist()

                # Container cho kết quả
                results_container = st.container()

                with results_container:
                    st.subheader("⏳ Đang phân tích...")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results = []

                    # Phân tích từng email
                    for i, email in enumerate(emails):
                        try:
                            # Phân tích email
                            analysis = evaluator.evaluate_importance(email)

                            # Thêm kết quả
                            results.append({
                                "Nội dung": email,
                                "Đánh giá": analysis
                            })

                            # Cập nhật tiến trình
                            progress = (i + 1) / len(emails)
                            progress_bar.progress(progress)
                            status_text.text(f"Đã phân tích {i+1}/{len(emails)} email")

                        except Exception as e:
                            st.error(f"Lỗi khi phân tích email {i+1}: {str(e)}")
                            continue

                    # Hiển thị kết quả
                    if results:
                        st.subheader("📊 Kết quả phân tích")
                        results_df = pd.DataFrame(results)

                        # Tạo bảng với định dạng
                        st.dataframe(
                            results_df,
                            column_config={
                                "Nội dung": st.column_config.TextColumn(
                                    "Nội dung Email",
                                    width="medium"
                                ),
                                "Đánh giá": st.column_config.TextColumn(
                                    "Đánh giá",
                                    width="small"
                                )
                            }
                        )

                        # Tạo nút download
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Tải kết quả phân tích",
                            data=csv,
                            file_name="email_analysis_results.csv",
                            mime="text/csv"
                        )

                        # Hiển thị thống kê
                        st.subheader("📈 Thống kê")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Tổng số email đã phân tích", 
                                    len(results))

                        with col2:
                            important_count = sum(1 for r in results 
                                               if "quan trọng" in r["Đánh giá"].lower())
                            st.metric("Số email quan trọng", 
                                    important_count)

        except Exception as e:
            st.error(f"Lỗi khi đọc file: {str(e)}")

if __name__ == "__main__":
    main()