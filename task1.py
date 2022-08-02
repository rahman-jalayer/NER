import spacy
import neuralcoref
import pandas as pd
import csv
import os


def co_reference(text):
    coref_of_text = nlp(text)
    return coref_of_text._.coref_resolved


def NER(text):
    all_entities = list()
    locations = list()
    ner_nlp = nlp(text)
    for ent in ner_nlp.ents:
        if ent.label_ not in ['CARDINAL', 'DATE', 'MONEY', 'TIME', 'PERCENT', 'QUANTITY', 'ORDINAL',
                              'LANGUAGE', 'LAW']:
            all_entities.append({'text': ent.text.lower(), 'label': ent.label_})
        if ent.label_ == 'GPE':
            locations.append(ent.text.lower())
    return all_entities, set(locations)


def similarity_calculation(word1, word2):
    try:
        nlp_word1 = nlp(word1)
        nlp_word2 = nlp(word2)
        if len(word1.split()) > 1:
            nlp_word1 = list(nlp_word1.noun_chunks)
        if len(word2.split()) > 1:
            nlp_word2 = list(nlp_word2.noun_chunks)

        if nlp_word1[0].has_vector and nlp_word2[0].has_vector:
            return nlp_word1[0].similarity(nlp_word2[0])
        return 0
    except Exception as e:
        return 0


def belong_score_computation(all_entities, locations):
    related_scores = dict()
    for location in locations:
        location_score = 0
        for entity in all_entities:
            if entity['label'] == 'GPE' and entity['text'] != location:
                similarty = 0
            else:
                similarty = similarity_calculation(location, entity['text'])
            location_score += similarty
        related_scores[location] = location_score
    return related_scores


def compute_belong_percent(scores):
    total_score = 0
    for _, score in scores.items():
        total_score += score
    for location, score in scores.items():
        scores[location] = round((score / total_score) * 100, 1)
    return scores


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_lg')
    neuralcoref.add_to_pipe(nlp)
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "dataset", "news.csv"),
                     usecols=['ID', 'Description'])
    resluts = dict()
    for _, row in df.iterrows():
        text = row['Description']
        coref_added_text = co_reference(text)
        all_entities, locations = NER(text)
        beloging_scores = belong_score_computation(all_entities, locations)
        belong_percent = compute_belong_percent(beloging_scores)
        resluts[row['ID']] = belong_percent

    header = ['ID', 'Location name', 'Percentage']
    with open('prediction.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for id, result in resluts.items():
            for loc, percentage in result.items():
                row = [id, loc, percentage]
                writer.writerow(row)
