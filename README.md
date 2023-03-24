# text_classfication_model

  my_class = Classification()
  text = "การบริการดีสัดครับ"
  negative_score = my_class.predict_negative(text)
  possivite_score = my_class.predict_possitve(text)
  male_score = my_class.predict_male(text)
  female_score = my_class.predict_female(text)
  toxic_score = my_class.predict_toxic(text)
  all_score = my_class.predict_all(text)
