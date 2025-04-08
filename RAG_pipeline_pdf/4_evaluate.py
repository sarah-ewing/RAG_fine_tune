import functions as fun

#################################   
import pandas as pd

df = pd.read_csv("silver_data.csv")
df = df[(df['section']!= 'References') & (df['section']!= 'Appendix') & (df['section']!='Disclaimer')]
df = df.reset_index(drop=True)
df['context'] = df['contex']

df['question'] = df['question'].str.replace('\n', '')
df['answer'] = df['answer'].str.replace('\n', '')
df['context'] = df['context'].str.replace('\n', '')

df['similarity_score'] = 0.0
df['BLEU'] = 0.0
df['Cosine'] = 0.0

df = df[['section', 'title', 'file_name', 'document_type', 'page', 'total_pages', 'context', 'question', 'answer', 'similarity_score', 'BLEU', 'Cosine']]


#################################
for i in range(0, len(df['question'])):
    user_input = df['question'].iloc[i]
    response = df['answer'].iloc[i]
    retrieved_context = df['context'].iloc[i]

    similarity_score = fun.calculate_fuzzy_similarity(response, retrieved_context)
    df.loc[i, 'similarity_score'] = similarity_score
    # print(f"Fuzzy Similarity Score: {similarity_score}")

    tfidf_similarity = fun.calculate_tfidf_cosine_similarity(response, retrieved_context)
    df.loc[i, 'Cosine'] = tfidf_similarity
    # print(f"TF-IDF Cosine Similarity: {tfidf_similarity:.2f}")

    bleu_score = fun.calculate_bleu_score(response, retrieved_context)
    df.loc[i, 'BLEU'] = bleu_score
    # print(f"BLEU Score: {bleu_score:.2f}")

# print(df.head())

df.to_csv('silver_data_graded.csv', index=False)


### Half Golden
df = pd.read_csv("silver_data_graded.csv")
df = df[(df['similarity_score'] > 30) & (df['BLEU'] > 0.006) & (df['Cosine'] > 0.46)]


print('Half Golden:', df.shape)

df = df.reset_index(drop=True)
df[['section', 'title', 'file_name', 'document_type', 'page', 'total_pages', 'context', 'question', 'answer', 'similarity_score', 'BLEU', 'Cosine']].to_csv('half_golden.csv', index=False)