from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import pymysql
import re
import underthesea
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


app = Flask(__name__)
CORS(app)

# Cấu hình OpenAI API Key (thay YOUR_API_KEY bằng API key của bạn)
openai.api_key = ""

# 🔹 Kết nối MySQL
def connect_db():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="",
        database="chatbot_db",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )

# 🔹 So sánh chuỗi theo từng từ
def compare_strings_by_word(string1, string2):
    # Tiền xử lý từng chuỗi - chuyển thành chữ thường, loại bỏ dấu câu và tách từ
    words1 = set(preprocess_question(string1).split())  # sử dụng hàm preprocess_question
    words2 = set(preprocess_question(string2).split())

    # Tìm từ chung
    matching_words = words1.intersection(words2)

    # Tính tỷ lệ khớp
    total_words = len(words1.union(words2))
    matching_percentage = (len(matching_words) / total_words) * 100 if total_words > 0 else 0

    # Kết quả
    return {
        "matching_words": list(matching_words),
        "matching_percentage": round(matching_percentage, 2),
    }
# 🔹 Tiền xử lý câu hỏi người dùng
def preprocess_question(question):
    question = question.lower().strip()
    question = re.sub(r'[^\w\s]', '', question)  # Loại bỏ dấu câu
    question_tokens = underthesea.word_tokenize(question)  # Tách từ tiếng Việt
    return " ".join(question_tokens)

# 🔹 Lấy danh sách intents từ database
def get_all_intents():
    db = connect_db()
    try:
        with db.cursor() as cursor:
            cursor.execute("SELECT id, intent_name FROM intents")
            intents = {row["id"]: row["intent_name"] for row in cursor.fetchall()}
        return intents  # Trả về danh sách intents {id: tên intent}
    finally:
        db.close()


# 🔹 Xác định intent của câu hỏi
# 🔹 Xác định intent của câu hỏi
def find_intent(question):
    db = connect_db()
    try:
        with db.cursor() as cursor:
            # Lấy từ khóa cùng intent_id và response_id từ DB
            cursor.execute("""
                SELECT keyword, intent_id, response_id
                FROM keywords_responses
            """)
            keywords = cursor.fetchall()  # [{keyword, intent_id, response_id}, {...}]

        # Sắp xếp từ khóa theo độ dài giảm dần
        keywords = sorted(keywords, key=lambda k: len(k['keyword']), reverse=True)

        # Tiền xử lý câu hỏi người dùng
        processed_question = preprocess_question(question)

        # 1. Kiểm tra từ khóa "chính xác" trong câu hỏi
        for keyword_data in keywords:
            keyword = preprocess_question(keyword_data['keyword'])
            intent_id = keyword_data['intent_id']
            response_id = keyword_data['response_id']

            # Nếu keyword xuất hiện chính xác (bao gồm cả chuỗi con), trả về intent + response
            if keyword in processed_question:
                return {"intent_id": intent_id, "response_id": response_id}

        # 2. Nếu không có keyword nào khớp chính xác, sử dụng Fuzzy Matching
        best_match = {"intent_id": None, "response_id": None}  # Kết quả tốt nhất
        best_match_score = 0

        for keyword_data in keywords:
            keyword = preprocess_question(keyword_data['keyword'])
            intent_id = keyword_data['intent_id']
            response_id = keyword_data['response_id']

            # Tính độ tương đồng fuzzy
            similarity_score = fuzz.token_set_ratio(processed_question, keyword)

            if similarity_score > best_match_score:
                best_match_score = similarity_score
                best_match = {"intent_id": intent_id, "response_id": response_id}

        # 3. Nếu Fuzzy Matching đạt ngưỡng 70, chọn intent phù hợp
        if best_match_score >= 85:
            return best_match

        # Nếu không tìm thấy kết quả phù hợp
        return {"intent_id": None, "response_id": None}

    finally:
        db.close()

# 🔹 Lấy câu trả lời từ database
def get_response_from_db(intent_id, response_id):
    print("Debug: intent_id =", intent_id, ", response_id =", response_id)

    if not intent_id or not response_id:
        return "Xin lỗi, tôi không có phản hồi phù hợp."  # Trường hợp không đủ thông tin.

    db = connect_db()
    try:
        with db.cursor() as cursor:
            # Truy vấn bằng cả intent_id và response_id
            cursor.execute("""
                SELECT response_text
                FROM responses
                WHERE intent_id = %s AND id = %s
            """, (intent_id, response_id))

            result = cursor.fetchone()
            return result['response_text'] if result else "Xin lỗi, không tìm thấy phản hồi phù hợp."
    finally:
        db.close()


# 🔹 Gọi OpenAI ChatGPT API
def get_chatgpt_response(question):

    try:
        question_with_uneti = f"[Trường đại học kinh tế-kỹ thuật công nghiệp(UNETI): ] {question}"
        response = openai.ChatCompletion.create(  # <--- Đúng với SDK >= 1.0.0
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Bạn là một chatbot hỗ trợ người dùng."},
                {"role": "user", "content": question_with_uneti},
            ],
            temperature=0.7,  # Điều chỉnh mức độ sáng tạo (0.0 thấp nhất, 2.0 cao nhất)
            max_tokens=300    # Giới hạn số lượng từ trả về
        )
        # Trả về nội dung của phản hồi
        return response.choices[0].message.get("content").strip()
    except Exception as e:
        print(f"Lỗi khi gọi OpenAI API: {e}")
        return "Xin lỗi, tôi không thể trả lời câu hỏi của bạn vào lúc này."


# 🔹 API xử lý câu hỏi
@app.route('/get-answer', methods=['POST'])
def get_answer():
    # Lấy câu hỏi từ người dùng
    user_data = request.get_json()
    question = user_data.get("question", "").strip()

    if not question:  # Kiểm tra câu hỏi có hợp lệ không
        return jsonify({
            "intent_id": None,
            "response_id": None,
            "answer": "Vui lòng nhập câu hỏi hợp lệ."
        }), 400

    # Gọi `find_intent` để xác định intent và response
    result = find_intent(question)
    intent_id = result.get("intent_id")
    response_id = result.get("response_id")

    # Nếu có intent_id và response_id, lấy câu trả lời từ cơ sở dữ liệu
    if intent_id and response_id:
        response = get_response_from_db(intent_id, response_id)
        return jsonify({
            "intent_id": intent_id,
            "response_id": response_id,
            "answer": response
        })

    # 4. Nếu không có câu trả lời, sử dụng ChatGPT
    gpt_response = get_chatgpt_response(question)
    return jsonify({"answer": gpt_response})

# Chạy server Flask
if __name__ == '__main__':
    app.run(debug=True)
