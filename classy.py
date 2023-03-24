import re
import pickle
from pythainlp.util import dict_trie
from pythainlp.corpus.common import thai_words
from pythainlp.tokenize import Tokenizer, word_tokenize

def split_fn(x):
    return x.split(' ')

class Classification(object):
    def __init__(self):
        self.custom_tokenizer = self.custom_token()

        self.enc_possitive = pickle.load(open('./text_classfication_model/encoder_content_possitive.pickle', 'rb'))
        self.norm_possitive = pickle.load(open('./text_classfication_model/normalizer_content_possitive.pickle', 'rb'))
        self.positive_model = pickle.load(open('./text_classfication_model/LinearSVC_possivite.pickle', 'rb'))

        self.enc_negative = pickle.load(open('./text_classfication_model/encoder_negative.pickle', 'rb'))
        self.norm_negative = pickle.load(open('./text_classfication_model/normalizer_negative.pickle', 'rb'))
        self.negative_model = pickle.load(open('./text_classfication_model/LinearSVC_negative.pickle', 'rb'))

        self.enc_female = pickle.load(open('./text_classfication_model/encoder_female.pickle', 'rb'))
        self.norm_female = pickle.load(open('./text_classfication_model/normalizer_female.pickle', 'rb'))
        self.female_model = pickle.load(open('./text_classfication_model/LinearSVC_female.pickle', 'rb'))

        self.enc_male = pickle.load(open('./text_classfication_model/male_encoder_content.pickle', 'rb'))
        self.norm_male = pickle.load(open('./text_classfication_model/male_normalizer_content.pickle', 'rb'))
        self.male_model = pickle.load(open('./text_classfication_model/linearSVC_male_classification.pickle', 'rb'))

        self.enc_toxic = pickle.load(open('./text_classfication_model/encoder_content_toxic.pickle', 'rb'))
        self.norm_toxic = pickle.load(open('./text_classfication_model/normalizer_content_toxic.pickle', 'rb'))
        self.toxic_model = pickle.load(open('./text_classfication_model/LinearSVC_content_toxic.pickle', 'rb'))
        text = None

    def custom_token(self):

        words = ['เลยค้าบ','ฮาฟ','ครับผม','ดีสัส','เอ็ง','โครตหล่อ','เย็ดเข้','รำคาญ','ดีออก','ครัฟ','ไม่น่ารัก','เหงา','ดีค้าบ','ไม่หล่อ',
                 'ครัช','จร่า','โครตเจ๋ง','จ้ะ','หวัดดีครับ','ได้ครับ','คะ','ค่ะ','ไม่พอใจ','สวัสดีค้าบ','ไม่ถูกใจ','จ่ะ','ไม่แย่','ขอบคุณครับ',
                 'เครียด','สวยเหี้ย','จร้า','ดีเหี้ย','คัช','โคตร','ข้าพระพุทธเจ้า','ดีสาส','เบื่อ','น่าเบื่อ','กลัว','เจ็บใจ','น่ากลัว','กระผม',
                 'ฮับ','คระ','คัฟ','งับ','ห่วย','อะไรนะค้าบ','เหยดโด้','โครตดี','โมโห','เลยครับ','คร้ะะ','สวยสาส','จ๊ะ','หดหู่','เหยดเป็ด',
                 'ค้ะ','ไม่ชอบ','เดี้ยน','เกลียด','คร้า','สวัสดีครับ','โครตเก่ง','ขอบคุณค้าบ','คร่ะ','สวยค้าบ','เหยด','อาตมา','เหยดเข้','ข้า',
                 'เท่ห์สาส','งอล','ดีสัด','โครตโหด','คับ','อะไรนะครับ','จ๋า','ได้ค้าบ','ไม่กลัว','หงุดหงิด','ผม','หวัดดีค้าบ','สวยครับ','ค๋า',
                 'อย่างแจ่ม','เศร้า','น่ารำคาญ','ดีครับ','ไม่สบายใจ','ขอขอบคุณค้าบ','หนู','งอน','สวยสัส','ดิฉัน','เสียใจ','เกล้ากระหม่อม',
                 'หม่อมฉัน','คร้ะ','อึดอัด','คร่า','ขอขอบคุณครับ','โกรธ','ครับ','โครต']

        custom_words_list = set(thai_words())
        custom_words_list.update(words)
        trie = dict_trie(dict_source=custom_words_list)
        custom_tokenizer = Tokenizer(custom_dict=trie, engine='newmm')
        return custom_tokenizer
    
    def clean_text(self, text):
        text = re.sub("\s+", "", text)
        new_texts = ' '.join([word for word in self.custom_tokenizer.word_tokenize(text)])
        return new_texts

    def predict_possitve(self, text):
        new_texts = self.clean_text(text)
        new_text_bow = self.enc_possitive.transform([new_texts])
        new_text_normalized = self.norm_possitive.transform(new_text_bow)
        prediction = self.positive_model.predict(new_text_normalized)[0]
        prediction_score = self.positive_model._predict_proba_lr(new_text_normalized)[:,1]
        return prediction, prediction_score[0]

    def predict_negative(self, text):
        new_texts = self.clean_text(text)
        new_text_bow = self.enc_negative.transform([new_texts])
        new_text_normalized = self.norm_negative.transform(new_text_bow)
        prediction = self.negative_model.predict(new_text_normalized)[0]
        prediction_score = self.negative_model._predict_proba_lr(new_text_normalized)[:,1]
        return prediction, prediction_score[0]
    
    def predict_male(self, text):
        new_texts = self.clean_text(text)
        new_text_bow = self.enc_male.transform([new_texts])
        new_text_normalized = self.norm_male.transform(new_text_bow)
        prediction = self.male_model.predict(new_text_normalized)[0]
        prediction_score = self.male_model._predict_proba_lr(new_text_normalized)[:,1]
        return prediction, prediction_score[0]

    def predict_female(self, text):
        new_texts = self.clean_text(text)
        new_text_bow = self.enc_female.transform([new_texts])
        new_text_normalized = self.norm_female.transform(new_text_bow)
        prediction = self.female_model.predict(new_text_normalized)[0]
        prediction_score = self.female_model._predict_proba_lr(new_text_normalized)[:,1]
        return prediction, prediction_score[0]
    
    def predict_toxic(self, text):
        new_texts = self.clean_text(text)
        new_text_bow = self.enc_toxic.transform([new_texts])
        new_text_normalized = self.norm_toxic.transform(new_text_bow)
        prediction = self.toxic_model.predict(new_text_normalized)[0]
        prediction_score = self.toxic_model._predict_proba_lr(new_text_normalized)[:,1]
        return prediction, prediction_score[0]
    
    def predict_all(self, text):
        pre_p, sc_p = self.predict_possitve(text)
        pre_n, sc_n = self.predict_negative(text)
        pre_m, sc_m = self.predict_male(text)
        pre_f, sc_f = self.predict_female(text)
        pre_t, sc_t = self.predict_toxic(text)

        return {'text' :text,
                'possitve': {pre_p:sc_p },
                'negative': {pre_n:sc_n},
                'male'    : {pre_m:sc_m},
                'female'  : {pre_f:sc_f},
                'toxic'   : {pre_t:sc_t}
                }
