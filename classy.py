import re
import pickle
from pythainlp.util import dict_trie
from pythainlp.corpus.common import thai_words
from pythainlp.tokenize import Tokenizer, word_tokenize
from functools import reduce

def split_fn(x):
    return x.split(' ')

class Classification(object):
    def __init__(self, df):
        self.custom_tokenizer = self.custom_token()
        self.df = df
        self.df["token"] = self.df["message"].apply(self.clean_text)

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
    
    def clean_text(self,text):
        return ' '.join(self.custom_tokenizer.word_tokenize(re.sub('[^\u0E00-\u0E7Fa-zA-Z0-9]+', '', text)))

    def predict_possitve(self):
        mfeat = self.enc_possitive.transform(self.df['token'].values)
        normalized = self.norm_possitive.transform(mfeat)
        self.df['possitve_type'] = self.positive_model.predict(normalized)
        self.df['possitve_score'] = self.positive_model._predict_proba_lr(mfeat)[:,1]
        return self.df[['message','possitve_type', 'possitve_score']]

    def predict_negative(self):
        mfeat = self.enc_negative.transform(self.df['token'].values)
        normalized = self.norm_negative.transform(mfeat)
        self.df['negative_type'] = self.negative_model.predict(normalized)
        self.df['negativen_score'] = self.negative_model._predict_proba_lr(mfeat)[:,1]
        return self.df[['message','negative_type', 'negativen_score']]
    
    def predict_male(self):
        mfeat = self.enc_male.transform(self.df['token'].values)
        normalized = self.norm_male.transform(mfeat)
        self.df['male_type'] = self.male_model.predict(normalized)
        self.df['male_score'] = self.male_model._predict_proba_lr(mfeat)[:,1]
        return self.df[['message','male_type', 'male_score']]

    def predict_female(self):
        mfeat =  self.enc_female.transform(self.df['token'].values)
        normalized = self.norm_female.transform(mfeat)
        self.df['female_type'] = self.female_model.predict(normalized)
        self.df['female_score'] = self.female_model._predict_proba_lr(mfeat)[:,1]
        return self.df[['message','female_type', 'female_score']]
    
    def predict_toxic(self):
        mfeat = self.enc_toxic.transform(self.df['token'].values)
        normalized = self.norm_toxic.transform(mfeat)
        self.df['toxic_type'] = self.toxic_model.predict(normalized)
        self.df['toxic_score'] = self.toxic_model._predict_proba_lr(mfeat)[:,1]
        return self.df[['message','toxic_type', 'toxic_score']]
    
    def predict_all(self):
        self.predict_possitve()
        self.predict_negative()
        self.predict_male()
        self.predict_female()
        self.predict_toxic()
        return self.df
