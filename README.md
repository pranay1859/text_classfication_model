# Text Classification Model

Here's an example of how to use the `Classification` class to predict scores for different categories based on a given text:

```python
from text_classfication_model.classy import split_fn, Classification

msg_test = pd.read_csv("msg_df_nir.csv")
dump_df = msg_test.head(100000)
dump_df = dump_df.convert_dtypes()
dump_df.message.fillna("", inplace=True)

# Prediction all
my_class = Classification(dump_df)
all_score = my_class.predict_all()

# Prediction one col
my_class = Classification(dump_df)
predict_female_df = my_class.predict_female()

```
