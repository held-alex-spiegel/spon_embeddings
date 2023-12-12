import pandas as pd
import tiktoken
import numpy as np
from openai import OpenAI
from scipy.spatial.distance import cosine
import config

client = OpenAI(config.OPENAI_API_KEY)


def update_embeddings(sas_url):
    
    # Read saved data
    df_saved = pd.read_csv(sas_url, sep=";")
    df_saved['publish_date'] = pd.to_datetime(df_saved['publish_date'], unit='s')

    # Load existing embeddings
    spon_embeddings = pd.read_csv('./data/spon_embeddings.csv')
    spon_embeddings_max_date = spon_embeddings['publish_date'].max()

    # Filter for new articles
    df = df_saved[df_saved['publish_date'] > spon_embeddings_max_date]
    
    # if df is empty, return with message
    if len(df) == 0:
        return print(f'No new articles found. Max date in embedding data is {spon_embeddings_max_date}')

    # Combine text fields
    df["combined"] = ("title: " + df.title.str.strip() + "; heading: " + df.heading.str.strip() + "; intro: " + df.intro.str.strip())

    # Tokenization and filtering based on token count
    embedding_encoding = "cl100k_base"
    max_tokens = 8000
    encoding = tiktoken.get_encoding(embedding_encoding)
    df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))
    df = df[df.n_tokens <= max_tokens]

    # Log the update details
    print(f'Max date in embedding data is {spon_embeddings_max_date} --> Need embeddings for {len(df)} new articles. Cost for {sum(df.n_tokens)} tokens will be approx. ${sum(df.n_tokens) * 0.0000001}')

    # Generate embeddings for new articles
    df['embedding'] = df.combined.apply(lambda x: client.embeddings.create(input=x, model="text-embedding-ada-002").data[0].embedding)

    # Update the embeddings dataset
    spon_embeddings.to_csv('./data/spon_embeddings_backup.csv', index=False)
    spon_embeddings = pd.concat([spon_embeddings, df], ignore_index=True)
    spon_embeddings = spon_embeddings.drop_duplicates(subset=['id'])
    spon_embeddings.to_csv('./data/spon_embeddings.csv', index=False)

    return print(f'Embeddings updated.')



def find_similar_articles(embedding_space, article_id=None, avg_embedding=None):
    pd.options.mode.chained_assignment = None
    
    if article_id is not None:
        embedding_index = embedding_space[embedding_space['id'] == article_id].index.values.astype(int)[0]
        single_embedding = embedding_space.iloc[embedding_index].embedding
        
    if avg_embedding is not None:
        single_embedding = avg_embedding

    appended_data = []
    for i in list(range(len(embedding_space)-1)):
        df_sim = cosine(single_embedding, embedding_space.iloc[1+i].embedding)
        data = embedding_space.iloc[1+i]
        data['cosine_similarity'] = df_sim
        
        appended_data.append(data)
    df_result = pd.DataFrame(appended_data)
    df_result = df_result.sort_values(by=['cosine_similarity'], ascending=False)
    
    return df_result



def get_average_embedding(embedding_space, article_id_list):
    
    filtered_embedding_space = embedding_space[embedding_space.id.isin(article_id_list)]
    filtered_embedding_space['UserId'] = 1
    average_user_embeddings = filtered_embedding_space.groupby('UserId')['embedding'].apply(lambda x: np.mean(np.stack(x), axis=0))
    
    return average_user_embeddings




def analyze_clusters(filtered_spon_embeddings, rev_per_cluster=5, print_output=True):
    n_clusters = len(filtered_spon_embeddings.Cluster.value_counts())

    repsonse_df= []
    
    for i in range(n_clusters):
        if print_output:
            print(f"Cluster {i}:", end=" ")

        # Selecting reviews from the cluster
        reviews = "\n".join(
            filtered_spon_embeddings[filtered_spon_embeddings.Cluster == i]
            .sort_values("distance_from_center", ascending=True).head(rev_per_cluster)
            .combined.str.replace("Title: ", "")
            .str.replace("\n\nContent: ", ":  ")
            .values
        )

        # Generating a prompt for the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to find common topics for multiple news articles. The response should be a 1-3 words (topic labels) that describes the similarities between the given articles."},
                {"role": "user", "content": f'Was haben die folgenden Nachrichten-Artikel gemeinsam? \n\nNachrichten-Artikel:\n"""\n{reviews}\n"""\n'},
                ],
            temperature=0,
            max_tokens=64,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        # Print the response
        if print_output:
            print(response.choices[0].message.content.replace("\n", ""))

            # Print titles and headings from the cluster sample
            sample_cluster_rows = filtered_spon_embeddings[filtered_spon_embeddings.Cluster == i].sort_values("distance_from_center").head(rev_per_cluster)#.sample(rev_per_cluster, random_state=42)
            for j in range(rev_per_cluster):
                print(sample_cluster_rows.title.values[j], end=":   ")
                print(sample_cluster_rows.heading.str[:70].values[j])

            print("-" * 100)
        
        # create dataframe with cluster and response
        cluster_df = {'Cluster': i, 'Response': response.choices[0].message.content.replace("\n", "")}
        cluster_df = pd.DataFrame(cluster_df, index=[0])
        repsonse_df.append(cluster_df)
    
    repsonse_df = pd.concat(repsonse_df)
        
    return repsonse_df

