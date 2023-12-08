from transformers import BertTokenizer
import torch
import numpy as np
import speech_recognition as sr


def get_tag_values(data):

    tag_values = list(set(data["Tag"].values))
    tag_values.append("PAD")
    print(tag_values)
    tag2idx = {t: i for i, t in enumerate(tag_values)}
    return tag_values, tag2idx


def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


def predict_entities(text, model_path):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    tokenized_sentence = tokenizer.encode(text)
    input_ids = torch.tensor([tokenized_sentence]).cpu()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model = torch.load(model_path, map_location=device)
    tag_values = ['B-art', 'I-art', 'B-eve', 'I-eve', 'O', 'I-geo', 'I-per', 'I-gpe', 'B-gpe', 'I-nat',
                  'B-per', 'I-org', 'I-tim', 'B-nat', 'B-tim', 'B-org', 'B-geo', "PAD"]

    #data = pd.read_csv("Dataset/NER dataset.csv", encoding="latin1").fillna(method="ffill")
    #tag_values, tag2idx = get_tag_values(data)

    with torch.no_grad():
        output = loaded_model(input_ids)
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)

    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices[0]):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(tag_values[label_idx])
            new_tokens.append(token)

    label_mapping = {
        'B-art': 'Beginning of an art entity',
        'B-eve': 'Beginning of an event entity',
        'B-geo': 'Beginning of a geographic entity',
        'B-gpe': 'Beginning of a geopolitical entity',
        'B-nat': 'Beginning of a natural entity',
        'B-org': 'Beginning of an organization entity',
        'B-per': 'Beginning of a person entity',
        'B-tim': 'Beginning of a time entity',
        'I-art': 'Continuation of an art entity',
        'I-eve': 'Continuation of an event entity',
        'I-geo': 'Continuation of a geographic entity',
        'I-gpe': 'Continuation of a geopolitical entity',
        'I-nat': 'Continuation of a natural entity',
        'I-org': 'Continuation of an organization entity',
        'I-per': 'Continuation of a person entity',
        'I-tim': 'Continuation of a time entity',
        'O': 'Outside of any entity'
    }
    #print(new_tokens)
    print()

    print("{:<40}\t{}".format("Full Label", "Token"))
    print("-" * 60)

    for token, label in zip(new_tokens[1:], new_labels[1:]):
        if label != 'O':
            full_label = label_mapping.get(label, label)
            print("{:<40}\t{}".format(full_label, token))
    return full_label, token

def audio_to_text(audio_file_path):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)

        try:
          # Use Google Web Speech API
          transcribed_text = recognizer.recognize_google(audio_data)
          print("Transcribed Text:", transcribed_text)
          return transcribed_text
        except sr.UnknownValueError:
          print("Sorry, I couldn't understand the audio.")
          return None
        except sr.RequestError as e:
          print(f"Could not request results from Google Speech Recognition service; {e}")
          return None

    return text