import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np

from preprocessing import preprocess_text, create_dataset_from_files, filter_columns, remove_indexes
from flask import Flask, request, jsonify
from flask_cors import CORS

encoder_model = tf.keras.models.load_model("encoder.keras")
decoder_model = tf.keras.models.load_model("decoder.keras")

app = Flask(__name__)
CORS(app,origins='*')


def decode_sequence_seq2seq_model_with_bidirectional_lstm(input_sequence, encoder_model, decoder_model):
    # Encode the input as state vectors.
    e_out, *state_values = encoder_model.predict(input_sequence)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    # Populate the first word of target sequence with the start word.
    start_token = 'sostok'
    end_token = 'eostok'
    target_seq[0, 0] = target_word_index[start_token]

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, *decoder_states = decoder_model.predict(
            [target_seq] + [e_out] + state_values
        )

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])  # Greedy Search
        sampled_token = reverse_target_word_index[sampled_token_index + 1]

        if sampled_token != end_token:
            decoded_sentence += ' ' + sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == end_token) or (len(decoded_sentence.split()) >= (max_len_summary - 1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        state_values = decoder_states

    return decoded_sentence

def predict_text(text, encoder_model, decoder_model):
    original_text = text
    text = preprocess_text([text])
    text_list = original_text.split()

    if len(text_list) <= max_len_paragraphs:
        # text = expand_contractions(text)
        text = preprocess_text(text)
        start_token = 'sostok'
        end_token = 'eostok'
        text = f'{start_token} {text} {end_token}'

        seq = x_tokenizer.texts_to_sequences([' '.join(text_list)])
        padded = pad_sequences(seq, maxlen=max_len_paragraphs, padding='post')
        pred_summary = decode_sequence_seq2seq_model_with_bidirectional_lstm(
            padded.reshape(1, max_len_paragraphs), encoder_model, decoder_model
        )
        return pred_summary
    else:
        pred_summary = ''

        while len(text_list) % max_len_paragraphs == 0:
            text_list.append('')

        lst_i = max_len_paragraphs
        for i in range(0, len(text_list), max_len_paragraphs):
            _text_list = original_text.split()[i:i + lst_i]
            _text = ' '.join(_text_list)
            _text = ' '.join(
                _text.split()
            )

            # _text = expand_contractions(_text)
            _text = preprocess_text(_text)
            start_token = 'sostok'
            end_token = 'eostok'
            _text = f'{start_token} {_text} {end_token}'

            _seq = x_tokenizer.texts_to_sequences([_text])
            _padded = pad_sequences(_seq, maxlen=max_len_paragraphs, padding='post')
            _pred = decode_sequence_seq2seq_model_with_bidirectional_lstm(
                _padded.reshape(1, max_len_paragraphs), encoder_model, decoder_model
            )
            pred_summary += ' ' + ' '.join(_pred.split()[1:-2])
            pred_summary = ' '.join(pred_summary.split())

        return pred_summary

# Preprocess
train_files = ["train.01.jsonl"]
dev_files = ["dev.01.jsonl"]
test_files = ["test.01.jsonl"]

df_train = create_dataset_from_files(train_files)
df_dev = create_dataset_from_files(dev_files)
df_test = create_dataset_from_files(test_files)

df_train = filter_columns(df_train)
df_dev = filter_columns(df_dev)
df_test = filter_columns(df_test)

dataframes_to_preprocess = [df_train, df_dev, df_test]
for df in dataframes_to_preprocess:
    df['man_summary'] = df['id'] + ' ' + df['summary']

    df.drop(['id', 'summary'], axis=1, inplace=True)

for df in dataframes_to_preprocess:
    df['man_summary'] = df.apply(lambda row: preprocess_text(row['man_summary']), axis=1)

    df['paragraphs'] = df.apply(lambda row: preprocess_text(row['paragraphs']), axis=1)

df_train.man_summary = df_train.man_summary.apply(lambda x: f'_START_ {x} _END_')
df_dev.man_summary = df_dev.man_summary.apply(lambda x: f'_START_ {x} _END_')
df_test.man_summary = df_test.man_summary.apply(lambda x: f'_START_ {x} _END_')

start_token = 'sostok'
end_token = 'eostok'

df_train['man_summary'] = df_train['man_summary'].apply(lambda x: f'{start_token} {x} {end_token}')
df_dev['man_summary'] = df_dev['man_summary'].apply(lambda x: f'{start_token} {x} {end_token}')
df_test['man_summary'] = df_test['man_summary'].apply(lambda x: f'{start_token} {x} {end_token}')


max_len_paragraphs = max([len(text.split()) for text in df_train['paragraphs']] +
                   [len(text.split()) for text in df_dev['paragraphs']] +
                   [len(text.split()) for text in df_test['paragraphs']])

max_len_summary = max([len(text.split()) for text in df_train['man_summary']] +
                       [len(text.split()) for text in df_dev['man_summary']] +
                       [len(text.split()) for text in df_test['man_summary']])

x_train = df_train['paragraphs']
y_train = df_train['man_summary']

x_val = df_test['paragraphs']
y_val = df_test['man_summary']

x_tokenizer = Tokenizer()
x_tokenizer.fit_on_texts(list(x_train))

x_train_sequence = x_tokenizer.texts_to_sequences(x_train)
x_val_sequence = x_tokenizer.texts_to_sequences(x_val)

x_train_pad = pad_sequences(x_train_sequence, maxlen=max_len_paragraphs, padding='post')
x_val_pad = pad_sequences(x_val_sequence, maxlen=max_len_paragraphs, padding='post')

x_vocab_size = len(x_tokenizer.word_index) + 1

# Tokenizer for summary
y_tokenizer = Tokenizer()
y_tokenizer.fit_on_texts(list(y_train))

y_train_sequence = y_tokenizer.texts_to_sequences(y_train)
y_val_sequence = y_tokenizer.texts_to_sequences(y_val)

y_train_pad = pad_sequences(y_train_sequence, maxlen=max_len_summary, padding='post')
y_val_pad = pad_sequences(y_val_sequence, maxlen=max_len_summary, padding='post')

y_vocab_size = len(y_tokenizer.word_index) + 1

# Next, let’s build the dictionary to convert the index to word for target and source vocabulary:
reverse_target_word_index = y_tokenizer.index_word
reverse_source_word_index = x_tokenizer.index_word
target_word_index = y_tokenizer.word_index


@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    if "berita" in data:
        berita = data["berita"]
        return jsonify({"ringkasan": predict_text(berita,encoder_model,decoder_model)})
    else:
        return jsonify({"error": "Missing 'berita' key in request body"}), 400

if __name__ == '__main__':
    app.run(debug=False)
