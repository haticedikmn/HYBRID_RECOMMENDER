#İş Problemi
#ID'si verilen kullanıcı için item-based ve user-based recommender yöntemlerini kullanarak tahmin yapınız.

#Veri Seti Hikayesi
#Veri seti, bir film tavsiye hizmeti olan MovieLens tarafından sağlanmıştır.
#İçerisinde filmler ile birlikte bu filmlere yapılan derecelendirme puanlarını barındırmaktadır.
#27.278 filmde 2.000.0263 derecelendirme içermektedir.
#Bu veriler 138.493 kullanıcı tarafından 09 Ocak 1995 ile 31 Mart 2015
#tarihleri arasında oluşturulmuştur. Bu veri seti ise 17 Ekim 2016 tarihinde oluşturulmuştur.
#Kullanıcılar rastgele seçilmiştir. Seçilen tüm kullanıcıların en az 20 filme oy verdiği bilgisi mevcuttur.

#Değişkenler
#movie.csv
#movieId – Eşsiz film numarası. (UniqueID)
#title – Film adı

#rating.csv
#userid – Eşsiz kullanıcı numarası. (UniqueID)
#movieId – Eşsiz film numarası. (UniqueID)
#rating – Kullanıcı tarafından filme verilen puan
#timestamp – Değerlendirme tarihi

#Görev 1:
#Veri Hazırlama işlemlerini gerçekleştiriniz.
import pandas as pd

pd.set_option ('display.max_columns', 20)
pd.set_option ('display.width', None)


movie = pd.read_csv('/Users/melda/PycharmProjects/DSMLBC6/HAFTA_04/DERS_NOTLARI/movie.csv')
rating = pd.read_csv('/Users/melda/PycharmProjects/DSMLBC6/HAFTA_04/DERS_NOTLARI/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()
comment_counts = pd.DataFrame(df["title"].value_counts())
comment_counts.head()
rare_movies = comment_counts[comment_counts["title"] <= 1000].index  # datayı filtrelemek için
rare_movies[0:10]

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv(r'C:\Users\Casper\PycharmProjects\DataScience\WEEK04\movie.csv')
    rating = pd.read_csv(r'C:\Users\Casper\PycharmProjects\DataScience\WEEK04\rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df


user_movie_df = create_user_movie_df()

user_movie_df.head()
user_movie_df.shape

#Görev 2:
#Öneri yapılacak kullanıcının izlediği filmleri belirleyiniz.

random_user = 108170
random_user_df = user_movie_df[user_movie_df.index == random_user]
random_user_df.head()

random_user_df.notna().any()
random_user_df.columns[random_user_df.notna().any()]
type(random_user_df.columns[random_user_df.notna().any()])  #pandas.core.indexes.base.Index
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
movies_watched[0:10]

#Görev 3:
#Aynı filmleri izleyen diğer kullanıcıların verisine ve Id'lerine erişiniz.

len(movies_watched)
movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()

user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count.head()

user_movie_count = user_movie_count.reset_index()
user_movie_count.head()
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count.sort_values(by="movie_count", ascending=False)

perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
users_same_movies.head()

#Görev 4:
#Öneri yapılacak kullanıcı ile en benzer kullanıcıları belirleyiniz

movies_watched_df.head()
final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies.values)]
final_df.head()
final_df.shape

final_df[final_df.index == random_user]

final_df.T.head()
final_df.T.corr().head()
final_df.T.corr().unstack().head()
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.head()
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()
corr_df.head()

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values (by='corr', ascending=False)
top_users.head()

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

rating = pd.read_csv(r'C:\Users\Casper\PycharmProjects\DataScience\WEEK04\rating.csv')
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
top_users_ratings.head()

#Görev 5:
#Weighted Average Recommendation Score'u hesaplayınız ve ilk 5 filmi tutunuz
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.head()


top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df.head()
recommendation_df = recommendation_df.reset_index()


import matplotlib.pyplot as plt
recommendation_df["weighted_rating"].hist()
plt.show()

# weighted_rating'i 2.7'den büyük olanları getirelim:
recommendation_df[recommendation_df["weighted_rating"] > 2.7]
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 2.7].sort_values("weighted_rating", ascending = False)
movies_to_be_recommend.head()

movie = pd.read_csv ('/Users/melda/PycharmProjects/DSMLBC6/HAFTA_04/DERS_NOTLARI/movie.csv')
recommended_user_based_df = movies_to_be_recommend.merge (movie[["movieId", "title"]])
recommended_user_based_df.head()
recommended_user_based_df.shape

# Kullanıcının daha önce izlemediği filmlerin önerilmesi
recommended_user_based_df = recommended_user_based_df.loc[~recommended_user_based_df["title"].isin(movies_watched)][:5]

#Görev 6:
#Kullanıcının izlediği filmlerden en son en yüksek puan verdiği filmin adına göre item-based öneri yapınız.
#Görev 6:
#▪ 5 öneri user-based
#▪ 5 öneri item-based
#olacak şekilde 10 öneri yapınız.

user = 108170

movie = pd.read_csv (r'C:\Users\Casper\PycharmProjects\DataScience\WEEK04\movie.csv')
rating = pd.read_csv (r'C:\Users\Casper\PycharmProjects\DataScience\WEEK04\rating.csv')


# Öneri yapılacak kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınması:
movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)]. \
               sort_values (by="timestamp", ascending=False)["movieId"][0:1].values[0]

movie.loc[movie["movieId"] == movie_id, "title"]

user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]
user_movie_df.corrwith(movie).sort_values(ascending=False).head(5)

def item_based_recommender(movie_name, user_movie_df, head=10):
    movie = user_movie_df[movie_name]
    return user_movie_df.corrwith (movie).sort_values(ascending=False).head(head)


movies_from_item_based = item_based_recommender(movie[movie["movieId"] == movie_id]["title"].values[0], user_movie_df, 20).reset_index()
movies_from_item_based.head()
movies_from_item_based.rename(columns={0:"corr"}, inplace=True)
movies_from_item_based.head()

# Kullanıcının daha önce izlemediği filmlerin önerilmesi
recommended_item_based_df = movies_from_item_based.loc[~movies_from_item_based["title"].isin(movies_watched)][:5]
recommended_item_based_df

#                                    title      corr
# 1              My Science Project (1985)  0.570187
# 2                    Mediterraneo (1991)  0.538868
# 3        Old Man and the Sea, The (1958)  0.536192
# 4  National Lampoon's Senior Trip (1995)  0.533029
# 5                   Clockwatchers (1997)  0.483337


