from flask import Flask, jsonify, render_template, request
from flask_restful import Resource, Api
import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
# Load data
from keras.models import load_model
import json
model = load_model('data/model/modelGetfit_0623.h5')
intents = json.loads(open('data/intents/intents_Getfit_062023.json', encoding="utf8").read())
words = pickle.load(open('data/model/textsGetfit_0623.pkl','rb'))
classes = pickle.load(open('data/model/labelsGetfit_0623.pkl','rb'))

def transText(text_input, scr_input='user'):
    from googletrans import Translator
    # define a translate object
    translate = Translator()
    if scr_input == "bot":
        result = translate.translate(text_input, src='en', dest='vi')
        result = result.text
    elif scr_input == "user":
        result = translate.translate(text_input, src='vi', dest='en')
        result = result.text
    else:
        result = "We not support this language, please use English or Vietnamese!"
    return result

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    print(res)
    # AMBIGOUS_THRESHOLD = 0.0
    CERTAIN_THRESHOLD = 0.7
    results = [[i,r] for i,r in enumerate(res) if r>CERTAIN_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    # print(results)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = i['responses']
            break
    return result, tag

def chatbot_response(msg):
    ints = predict_class(msg, model)
    print(ints)
    if ints:
        res, tag = getResponse(ints, intents)
    else:
        res = ["Rất xin lỗi vì thông tin bạn cần không tồn tại trong hệ thống, chúng tôi sẽ kiểm tra và cập nhật trong thời gian tới. Bạn còn muốn biết thêm thông tin gì khác không?", "930e5fa5-827a-454f-bcac-84e1b9dd5b4f"]
        tag = "Other"
    return res, tag
def chat_rulebased_01(msg):
    if "nguyên tắc hoạt động" in msg.lower() or "tôn chỉ hoạt động" in msg.lower():
        res = ["Câu lạc bộ hoạt động theo tôn chỉ: 1. Thượng tôn pháp luật. 2. Tôn trọng ý kiến tập thể. 3. Trung thực & minh bạch rõ ràng. 4. Hiệu quả & kịp thời. 5. Hợp tác & vươn xa. 6. Thân thiện & bác ái.", "4c265861-cbd7-4863-8300-e0237c41e5b0"]
        tag = "CLB_Mission"
    elif "thành viên" in msg.lower():
        res = ["Câu lạc bộ tuyển thành viên theo những tiêu chí: 1. Những doanh nhân chủ doanh nghiệp các công ty đang hoạt động hợp pháp tại VN. 2. Là hội viên chính thức của CLB Doanh nhân Sài Gòn (có đóng phí thường niên theo quy định) 3. Yêu thích công tác thiện nguyện và các công tác cộng đồng khác. 4. Tôn trọng tôn chỉ hoạt động & quy định của Ban Công tác xã hội Doanh nhân Sài Gòn. 5. Cam kết không bàn luận các chủ đề liên quan đến tôn giáo, chính trị.", "f0890010-1a9c-4896-87a3-f0108087c6e9"]
        tag = "CLB_Recruit" 
    elif "đã thực hiện" in msg.lower():
        res = ["Những hoạt động ý nghĩa mà hội đã thực hiện trong thời gian gần đây là: trao tặng 7 căn nhà tình thương tại xã Trà Tây, huyện Trà Bồng, tỉnh Quảng Ngãi, hỗ trợ thuốc cho F0 tại nhà, dự án Việc làm trao tay đánh bay COVID-19, ...", "f2275ffa-57c0-43d8-a00d-63401d8a526a"]
        tag = "SW_activities" 
    elif "sẽ thực hiện" in msg.lower():
        res = ["Sắp tới hội sẽ tổ chức sự kiện Họp mặt đầu xuân nhằm tạo điều kiện để Quý anh chị doanh nhân có dịp Giao lưu - Kết nối - Thắt chặt tình đoàn kết với nhau, cũng như là dịp để Ban điều hành Ban Công tác xã hội chia sẻ kế hoạch và chương trình hành động của năm 2023", "43ee3e72-41be-4b48-98b5-a943026b1ae2"]
        tag = "SW_next_event" 
    else:
        res = ["Câu lạc bộ đã tổ chức nhiều hoạt động thiện nguyện trên hầu khắp các lĩnh vực như hỗ trợ đồng bào miền Trung, tiếp sức cán bộ y tế chống dịch Covid, hỗ trợ việc làm, cấp học bổng cho học sinh nghèo vượt khó, … ", "4c1f7f2c-6ac6-4a0f-aa45-f9179d3bcae4"]
        tag = "Introduce" 
    return res, tag
def chat_rulebased_02(msg):
    ### Function
    if "chức năng" in msg.lower() in msg.lower():
        res = ["Những chức năng cơ bản của Công tác xã hội là: 1. Chức năng phòng ngừa: nghiên cứu và dự báo xu hướng vận động của xã hội nhằm vận động, tư vấn để chính quyền có những chính sách phù hợp để ngăn ngừa sự phát sinh các vấn đề xã hội. 2. Chức năng chữa trị: chăm sóc sức khỏe, cải thiện tình hình kinh tế & việc làm, hạ tầng cơ sở, nước sạch vệ sinh môi trường, hỗ trợ tâm lý tình cảm… 3. Chức năng phục hồi: Giúp đỡ người không may gặp nạn vượt qua khó khăn và hòa nhập cộng đồng. 4. Chức năng phát triển: Là việc hỗ trợ để cho người gặp khó khăn có thể phát huy được những khả năng của bản thân vượt qua khó khăn để vươn lên tự lập trong cuộc sống.", "ae56a996-115f-417e-acbf-beb014d7c08f"]
        tag = "CLB_Function"
    
    ### Method
    elif "cá nhân và gia đình" in msg.lower():
        res = ["Công tác xã hội với trẻ em và gia đình là một phần trong các lĩnh vực chuyên biệt của ngành công tác xã hội với mục tiêu đem lại sự hỗ trợ cho trẻ em trong hoàn cảnh khó khăn, giúp bảo vệ trẻ em và gia đình và góp phần vào nền an sinh cho trẻ em và gia đình.", "06eed5d3-72c3-4536-a0af-a962cfeb2d44"]
        tag = "CLB_individualsAndFamilies"
    elif "nhóm phát triển cộng đồng" in msg.lower():
        res = ["Nhóm phát triển cộng đồng là những nhóm  hoạt động phi lợi nhuận, với tinh thần thiện nguyện hướng tới cung cấp các dịch vụ theo một quy trình chuyên môn chuyên nghiệp, mang lại hiệu quả cụ thể và lâu dài. Cam kết đồng hành với các cá nhân, gia đình và cộng đồng cùng hành động vượt qua những khó khăn, thách thức ở hiện tại nhằm hướng tới một cuộc sống khỏe mạnh, hạnh phúc và bình an.", "2bdee838-08e0-4eec-910b-a5e0a50613e2"]
        tag = "CLB_communityDevelopmentTeam"
    elif "quản trị" in msg.lower():
        res = ["Quản trị công tác xã hội là một phương pháp của công tác xã hội có liên quan đến việc cung ứng và phân phối các nguồn tài nguyên xã hội giúp con người đáp ứng nhu cầu của họ và phát huy tiềm năng bản thân.", "51c60243-d401-4341-93d8-9896a0b083da"]
        tag = "CLB_socialWorkAdministration"
    elif "phương pháp" in msg.lower():
        res = ["Các phương pháp của ngành Công tác xã hội bao gồm 2 nhóm phương pháp: 1. Nhóm phương pháp thực hành có Công tác xã hội với cá nhân và gia đình và Công tác xã hội với nhóm phát triển cộng đồng. 2. Nhóm phương pháp lý thuyết có quản trị Công tác xã hội và nghiên cứu trong Công tác xã hội", "36cb4786-cbd0-44d3-b132-703c1cd241f4"]
        tag = "CLB_Method"
    
    ### Research
    elif "nghiên cứu" in msg.lower():
        res = ["Nghiên cứu trong công tác xã hội được sử dụng bởi bản thân các nhân viên xã hội, các nhà làm chính sách, người cung cấp dịch vụ. Lý do là các nghiên cứu này xoay quanh việc thử nghiệm và khẳng định các chính sách và mô hình can thiệp xã hội là cần thiết cho đối tượng. Trên thực tế, một số mô hình can thiệp chưa hẳn đã tốt hơn cho đối tượng so với trước khi họ nhận được hỗ trợ.", "b2888587-8f2b-462e-8e02-062d23fbecb7"]
        tag = "CLB_SocialWork"
    
    ### Area
    elif "trẻ em" in msg.lower() or "phúc lợi xã hội" in msg.lower():
        res = ["Bảo vệ trẻ em và phúc lợi xã hội bao gồm những hoạt động: Quản lý trường hợp (hay quản lý đối tượng): một số trường hợp trẻ em gặp các vấn đề về bạo lực, xâm hại, bóc lột sức lao động, … Tham vấn, tư vấn hỗ trợ trẻ em và gia đình: Trong nhiều trường hợp trẻ em hoặc một số các thành viên trong gia đình gặp những khó khăn không thể tự giải quyết được và cần tới sự trợ giúp của nhân viên CTXH. Quản lý các chương trình tái hòa nhập gia đình: trẻ em vi phạm pháp luật; trẻ em nghiện ma túy; trẻ em bị mua bán; trẻ em mắc bệnh hiểm nghèo và phải điều trị dài ngày;… Sau đó trẻ em sẽ hòa nhập trở lại gia đình và cộng đồng nơi các em sinh sống. Quản lý và hỗ trợ các dịch vụ chăm sóc thay thế: bao gồm quá trình tiến hành đánh giá nhu cầu của trẻ, đánh giá gia đình nhận chăm sóc thay thế, gia đình nhận con nuôi, hỗ trợ tâm lý xã hội, tập huấn cho gia đình nhận nuôi, giám sát. Tham gia vận động chính sách: tuyên truyền, vận động, động viên,… Ngoài ra, có một số hoạt động thực hành khác có thể tham gia như: vận động nguồn lực để tăng cường dịch vụ hỗ trợ; thực hiện nghiên cứu các vấn đề xã hội và đưa ra đề xuất cải thiện chính sách xã hội.", "783ed9d5-1923-4dc7-9acd-70829388e32e"]
        tag = "CLB_childProtection"
    elif "y tế" in msg.lower():
        res = ["Trong lĩnh vực y tế chúng ta có các hoạt động: Hỗ trợ cho bệnh nhân và gia đình của họ giải quyết các vấn đề liên quan đến sức khỏe, bệnh tật hay tình trạng khuyết tật. Hỗ trợ tiếp cận các nguồn lực bên ngoài và điều kiện vật chất để chữa bệnh, phục hồi sức khỏe. Cùng với đội ngũ y bác sĩ, nhân viên CTXH sẽ tham gia vào quá trình chẩn đoán và điều trị bệnh liên quan đến tâm lý, xã hội; Chương trình phục hồi, tái hòa nhập cộng đồng sau khi bệnh nhân hoàn tất điều trị tại bệnh viện. Hỗ trợ các chương trình chăm sóc sức khỏe và phòng ngừa tại cộng đồng như bệnh tật theo mùa, kế hoạch hóa gia đình, sức khỏe sinh sản vị thành niên, giáo dục cha mẹ, HIV/AIDS, tai nạn thương tích.", "197a481d-2f3d-40e6-9ae0-c043887911c4"]
        tag = "CLB_medicalField"
    elif "tư pháp" in msg.lower():
        res = ["Lĩnh vực tư pháp có những hoạt động: Hỗ trợ nạn nhân và nhân chứng tham gia vào hệ thống tư pháp; Hỗ trợ nạn nhân là trẻ em và gia đình trong quá trình tham gia tố tụng; Thực hiện báo cáo xã hội cho tòa án; Hỗ trợ tâm lý-xã hội và phục hồi cho người chưa thành niên vi phạm pháp luật, những người được xử lý chuyển hướng; Hỗ trợ thực hiện các biện pháp cải tạo tại cộng đồng; Hỗ trợ phúc lợi cho những người trong trại giam.", "42fb1141-b362-48ef-8e02-740e8179fcbf"]
        tag = "CLB_legalField"
    elif "giáo dục" in msg.lower():
        res = ["Lĩnh vực giáo dục có những hoạt động như: Phát hiện các nguy cơ trong và ngoài cơ sở giáo dục có ảnh hưởng tiêu cực đến người học; phát hiện các vụ việc liên quan đến người học có hoàn cảnh đặc biệt, bị xâm hại, có hành vi bạo lực, bỏ học, vi phạm pháp luật; Tổ chức các hoạt động phòng ngừa, hạn chế nguy cơ người học rơi vào hoàn cảnh đặc biệt, bị xâm hại, bị bạo lực, bỏ học, vi phạm pháp luật; Thực hiện quy trình can thiệp, trợ giúp đối với người học có hoàn cảnh đặc biệt, bị xâm hại, bị bạo lực, bỏ học, vi phạm pháp luật;  Phối hợp với gia đình, chính quyền địa phương và các đơn vị cung cấp dịch vụ công tác xã hội tại cộng đồng, trợ giúp đối với người học cần can thiệp, trợ giúp khẩn cấp đối với giáo viên, người học có nhu cầu can thiệp và hỗ trợ; Tổ chức các hoạt động hỗ trợ phát triển, hòa nhập cộng đồng cho người học sau can thiệp và các trường hợp khác liên quan đến người học, giáo viên, phụ huynh khi có nhu cầu hỗ trợ.", "ac8b1fa3-0950-432d-8bab-7adc504a9e99"]
        tag = "CLB_fieldOfEducation"
    elif "lĩnh vực" in msg.lower():
        res = ["Một số lĩnh vực có thể thực hiện công tác xã hội: 1. Bảo vệ trẻ em và phúc lợi xã hội. 2. Y tế. 3. Tư pháp và 4. Giáo dục.", "8673b09b-2b23-485b-829b-26bf0aff0fac"]
        tag = "CLB_Areas"

    ### Definition
    else:
        res = ["Công tác xã hội là nghề thực hành và là một lĩnh vực học thuật hoạt động chuyên môn nhằm trợ giúp các cá nhân, nhóm, cộng đồng phục hồi hay tăng cường chức năng xã hội góp phần đảm bảo nền an sinh xã hội. Được thực hiện theo những nguyên tắc và được vận hành trên cơ sở văn hóa truyền thống của dân tộc nhằm giải quyết các nan đề trong cuộc sống của họ.", "788bed04-b6f1-4419-8dba-8ba8b7a8a711"]
        tag = "CLB_Definition"          
    return res, tag


app = Flask(__name__)
api = Api(app)



@app.route("/")
def home():
    return render_template("index.html")

@app.route('/welcome', methods=["POST"])
def voice_welcome():
    resp = "Getfit gym và Yoga xin kính chào quý khách, em là trợ lý ảo của câu lạc bộ, quý khách cần hỗ trợ gì ạ?"
    output = {
            "res_text": resp,
            "res_audio": "GetFit_welcome"
        }
    return jsonify(output)


class Chatbot(Resource):

    def post(self):
        text_input = request.get_json().get("message")
        if "ban công tác xã hội" in text_input.lower():
            resp, tag = chat_rulebased_01(text_input)
        elif "công tác xã hội" in text_input.lower():
            resp, tag = chat_rulebased_02(text_input)
        else:
            text_input = transText(text_input)
            try:
                resp, tag = chatbot_response(text_input)
            except:
                resp = ["Tín hiệu không ổn định, vui lòng lặp lại rõ hơn nhé", "fbad6e35-3933-4388-be7b-d6dda276e114"]
                tag = "Error"
            print(resp)
        output = {
            "res_text": resp[0],
            "audio_token": resp[1],
            "res_audio": tag
        }
        return jsonify(output)

api.add_resource(Chatbot, '/response')

if __name__ == "__main__":
    app.run(debug=True)