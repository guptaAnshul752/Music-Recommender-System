import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
import warnings
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
import recommendation as recommender
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

from nltk.tokenize import word_tokenize


from scipy.sparse import csr_matrix
warnings.filterwarnings('ignore')
sns.set()
pd.options.display.max_columns = 500
pd.options.display.max_rows = 500


##########################################################################################################################

###Initializing the dataset
df_triplets = pd.read_csv('recom/kaggle_visible_evaluation_triplets.txt',sep='\t', names=['user_id', 'song_id', 'listen_count'])
df_tracks = pd.read_csv('recom/unique_tracks.txt',sep='<SEP>', names=['track_id','song_id','artist_name','song'])
data = pd.merge(df_triplets, df_tracks.drop_duplicates(['song_id']), how='left', on='song_id')
song_info = data.sample(n=10000).reset_index(drop=True)
data.dropna(inplace=True)
data = data.head(900000)
data.drop(columns='track_id', axis=1, inplace=True)
users = data.groupby('user_id').apply(lambda x: dict(zip(x['song'],x['listen_count']))).to_dict()


##########################################################################################################################

class pop():
    
    def __init__(self):
        print("Welcome To popularity Based Filtering")
    
    def pop1(self):
        
#         train_data, test_data = train_test_split(data, test_size = 0.20, random_state=0)
        popularity_model= recommender.popularity_recommender_system()
        popularity_model.create(data,'user_id','song',10)
        p = popularity_model.recommend(data['user_id'][5])
        p.drop(['user_id'],axis=1,inplace=True)
        
        return p
        
    
##########################################################################################################################

class svd():
    
    def __init__(self):
        print("Welcome to SVD Filtering method")
    
    def get_user(self): 
        print("inside get user")
        # Get number of songs each user has listened
        song_user = data.groupby('user_id')['song_id'].count()
        # Get users which have listen to at least 10 songs
        song_ten_id = song_user[song_user > 30].index.to_list()
        # Filtered the dataset to keep only those users with more than 10 listened
        df_song_id_more_ten = data[data['user_id'].isin(song_ten_id)].reset_index(drop=True)
        return self.pv(df_song_id_more_ten)
        
        
        
    def pv(self,df_song_id_more_ten):    
        print("into pv")
        print(df_song_id_more_ten.shape)
        df_songs_features = df_song_id_more_ten.pivot(index='user_id', columns='song_id', values='listen_count').fillna(0)
        # obtain a sparse matrix
        mat_songs_features = csr_matrix(df_songs_features.values)
        
        mat= df_songs_features.to_numpy()
        user_ratings_mean = np.mean(mat, axis = 1)
        R_demeaned = mat - user_ratings_mean.reshape(-1, 1)
        U, sigma, Vt = svds(R_demeaned, k = 1)
        sigma = np.diag(sigma)
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
        preds_df = pd.DataFrame(all_user_predicted_ratings, columns = df_songs_features.columns)
        preds_df['user_id']=df_songs_features.index
        first_column = preds_df.pop('user_id')
        preds_df.insert(0, 'user_id', first_column)
        return preds_df
        
        
        
        
        
    def recommend_song(self,preds_df,userID, song_df, original_ratings_df, num_recommendations=5):
        predictions_df=preds_df   
        # Get and sort the user's predictions
        if(preds_df.user_id[preds_df.user_id==userID].count()==1):
            user_row_number =int(predictions_df[predictions_df.user_id==userID].index[0]) # UserID starts at 1, not 0
            print(user_row_number)
            sorted_user_predictions = predictions_df.iloc[user_row_number,1:].sort_values(ascending=False)

            # Get the user's data and merge in the song information.
            user_data = original_ratings_df[original_ratings_df.user_id == (userID)]
            user_full = (user_data.merge(song_df, how = 'left', left_on = 'song_id', right_on = 'song_id').sort_values(['listen_count'], ascending=False))

            print ("User {0} has already rated {1} song".format(userID, user_full.shape[0]))
            print ("Recommending the highest {0} predicted ratings song not already rated".format(num_recommendations))

            # Recommend the highest predicted rating song that the user hasn't seen yet.
            recommendations = (song_df[~song_df['song_id'].isin(user_full['song_id'])].merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
                       left_on = 'song_id',
                       right_on = 'song_id').
                 rename(columns = {user_row_number: 'Predictions'}).
                 sort_values('Predictions', ascending = False).
                               iloc[:num_recommendations, :-1]
                              )
            
            
            return user_full, recommendations
        else:
            return "User Not Found"

##########################################################################################################################

class usf():

    def __init__(self):
        print("Welcome to user based filtering method")

     
    def unique_items(self):
        unique_items_list = []
        
        for person in users.keys():
            for items in users[person]:
                unique_items_list.append(items)            
                
        s = set(unique_items_list)
        unique_items_list = list(s)
        
        return unique_items_list


# custom function to create pearson correlation method 

    def pearson_correlation(self,person1,person2):
        both_listened = {}
        for i in users[person1]:
            if i in users[person2]:
                both_listened[i] = 1

        no_of_songs = len(both_listened)
        if no_of_songs == 0:
            return 0

        person1_preferences_sum = sum([users[person1][i] for i in both_listened])
        person2_preferences_sum = sum([users[person2][i] for i in both_listened])

        # Sum up the squares of preferences of each user
        person1_square_preferences_sum = sum([pow(users[person1][i], 2) for i in both_listened])
        person2_square_preferences_sum = sum([pow(users[person2][i], 2) for i in both_listened])

        # Sum up the product value of both preferences for each song
        product_sum_of_both_users = sum([users[person1][i] * users[person2][i] for i in both_listened])

        # Calculate the pearson score
        numerator_value = product_sum_of_both_users - (person1_preferences_sum * person2_preferences_sum / no_of_songs)
        denominator_value = sqrt((person1_square_preferences_sum - pow(person1_preferences_sum, 2) / no_of_songs) * (person2_square_preferences_sum - pow(person2_preferences_sum, 2) / no_of_songs))
        
        if denominator_value == 0:
            return 0
        else:
            r = numerator_value / denominator_value
            return r


    # custom function to check most similar users

    def most_similar_users(self,target_person,no_of_users):
        # Used list comprehension for finding pearson similarity between users
        scores = [(pearson_correlation(target_person,i),i) for i in users if i !=target_person]

        scores.sort(reverse=True)
        
        #return the scores between the target person & other persons
        return scores[0:no_of_users]



    def target_songs_to_users(self,target_person):
        listened_songs = []
        unique_list = self.unique_items()
        
        for songs in users[target_person]:
            listened_songs.append(songs)

        s = set(unique_list)
        ignored_songs = list(s.difference(listened_songs))
        a = len(ignored_songs)
        
        if a == 0:
            return 0
        return ignored_songs, listened_songs

    def recommendation_phase(self,person):
        # Gets recommendations for a person by using a weighted average of every other user's frequency
        totals = {}  # empty dictionary
        simSums = {} # empty dictionary
        for i in users.keys():
            # don't compare me to myself
            if i == person:
                continue
            sim = self.pearson_correlation(person, i)

            # ignore scores of zero or lower
            if sim <= 0:
                continue
            for item in users[i]:
                # only score songs I haven't listened yet
                if item not in users[person]:
                    # Similrity * score
                    totals.setdefault(item, 0)
                    totals[item] += users[i][item] * sim
                    # sum of similarities
                    simSums.setdefault(item, 0)
                    simSums[item] += sim
                    # Create the normalized list

        rankings = [(total / simSums[item], item) for item, total in totals.items()]
        rankings.sort(reverse=True)
        # returns the recommended songs
        recommendataions_list = [(recommend_item,score) for score, recommend_item in rankings]
        return recommendataions_list

    def dis_user(self,tp):
        if tp in users.keys():
            lst = []
            a = self.recommendation_phase(tp)
            if a != -1:
                print("\n Recommendation using User-based Collaborative Filtering:  \n")
                count = 0
                for songs,weights in a:
                    if count == 10:
                        break
#                     print(songs,'(',weights,')')
                    count += 1
                    lst.append([songs, weights])
                    
            return lst
        
        else:
            a = 'No songs to suggest, might be new user \n Switch to the Popularity Recommendation Engine . . . !'
            print("Person not found ! \n")
            print(a)
            
##########################################################################################################################
            
class item_based():
    
    def __init__(self):
        print("Item-Item Based Collaborative Filtering...")
    
    def item_pred(self, lst):
        song_data=data.sample(n=50000).reset_index()
        item_recommender = recommender.item_similarity_recommender_py()
        item_recommender.create(song_data, 'user_id', 'song')
        user_items = item_recommender.get_user_items(song_data['user_id'][2])
        # display user songs history
        for user_item in user_items:
            print(user_item)
        item_recommender.recommend(song_data['user_id'][2])
        similar_items = item_recommender.get_similar_items(lst)
        
        return similar_items
        
#         return recommend_similar_item
#         print(item_recommender.get_similar_items(['Let Me Be ThE One', 'Carrera']))
        
        
##############################################################################################################################

class content():
    
    def __init__(self):
        print("Content Based Collaborative Filtering...")
        
    def _print_message(self, song, recom_song):
        rec_items = len(recom_song)
        content_lst = []
        
        print(f'The Top {rec_items} recommended songs for {song} are: \n')

        for i in range(rec_items):

#             print(f"Number {i+1}:")

#             print(f"{recom_song[i][1]} by {recom_song[i][2]} ( similarity score : {round(recom_song[i][0], 3)}  )") 
            content_lst.append([recom_song[i][1], recom_song[i][2], round(recom_song[i][0], 3)])
        
        return content_lst

        
    def recommend(self, recommendation):

        # Get song to find recommendations for
        song = recommendation['song']

        # Get number of songs to recommend
        number_songs = recommendation['number_songs']

        # Get the number of songs most similars from matrix similarities
        recom_song = self.matrix_similar[song][:number_songs]
        
        # print each item
        res = self._print_message(song=song, recom_song=recom_song)
        return res
        
    def initia(self):
        vectorizer = TfidfVectorizer(analyzer='word')
        return vectorizer
        
    def content_pred(self):
#         print(data.columns)
        cleaned_data = []
        
        for i in song_info['song'].iloc[:]:
            token =self.tokenizer(i)
            cleaned_data.append(token)
#         print(len(cleaned_data))
        song_matrix = self.initia().fit_transform(cleaned_data)
        cosine_similarities = cosine_similarity(song_matrix)
        print('Size of Cosine_Similarity Matrix :',cosine_similarities.shape)
        
        similarities = {}
        for i in range(len(cosine_similarities)):
                      
            # Now we'll sort each element in cosine_similarities and get the indexes of the songs. 
            similar_indices = cosine_similarities[i].argsort()[:-50:-1] 

            # After that, we'll store in similarities each name of the 50 most similar songs.
            # Except the first one that is the same song.

            similarities[data['song'].iloc[i]] = [(cosine_similarities[i][x], data['song'][x], data['artist_name'][x]) for x in similar_indices]

        self.matrix_similar=similarities
        return similarities

        
    def tokenizer(self,text):
        text = re.sub('<[^>]*>', '', str(text))
        emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
        text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
        tokenized = word_tokenize(text)
        word = ' '.join(tokenized)
        return word
    def song_listened(self,user):
        song = data[data['user_id'] == user].song.value_counts().keys()

        return (song)
    
    
##########################################################################################################################
    
class all(pop, usf,svd,item_based,content):
    def __init__(self):
        print("Welcome to MSD")
    

def main(user):

                    

    
    c = all()  
    target_person = user
    
    if target_person in users.keys():
        ignored_songs,listened_songs = c.target_songs_to_users(target_person)
        
        if len(listened_songs) > 30:
    
            print('Listened Songs :', len(listened_songs))
            print('Not Listened Songs :', len(ignored_songs),'\n')   

            dct = {"Listened songs": listened_songs}

            my_dict = pd.DataFrame(dct)
#             print(my_dict.head(10))

            # Pop songs
            p=c.pop1()
            popular = p.values.tolist()
            
            #p_columns = p.columns
            # print(p_columns)
            # print(popular)
            
    
    
            # User-based
            userl=c.dis_user(target_person)
            
    
    
            # # item-based
            item = c.item_pred(list(listened_songs))
            items = item.values.tolist()
            i_cols = item.columns
            # print(i_cols)
            # print(items)
            
    
    
    
            # #Content based
            similarities=c.content_pred()    
            recommendation = {
                "song": listened_songs[0],
                "number_songs": 5
            }
            content = c.recommend(recommendation)
            # c_cols = ['Song', 'Artist', 'Similarity Score']




# # #             Model based
            k = c.get_user()
            already_rated, predictions = c.recommend_song(k,target_person, df_tracks, df_triplets, 10)
            
            predictions.drop(['track_id','song_id'],axis=1,inplace=True)
            svd_pred = predictions.values.tolist()
#             svd_cols = predictions.columns
            # print(svd_pred)
            
            return popular,userl,items,svd_pred,content


        
        else:
            k = "Insufficient Data"
            print('As the User has listened < 30 songs, So we can only recommend Most Popular Songs to this User')
            p=c.pop1()
            popular = p.values.tolist()
            return popular,k,k,k,k
            
        
    else:
        print("Uesr not Found !!!")
        tr="user not found"
        return tr,tr,tr,tr,tr


