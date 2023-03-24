# Text Classification Model

Here's an example of how to use the `Classification` class to predict scores for different categories based on a given text:

```python
from classy import split_fn, Classification
my_class = Classification()
text = "การบริการดีสัดครับ"
negative_score = my_class.predict_negative(text)
positive_score = my_class.predict_possitve(text)
male_score = my_class.predict_male(text)
female_score = my_class.predict_female(text)
toxic_score = my_class.predict_toxic(text)
all_score = my_class.predict_all(text)
```
