import pandas as pd
import streamlit as st
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


st.set_page_config(layout='wide')

st.title('PF 2024 recommender')

# newsfolder = 'C:/Users/matth/PycharmProjects/Inframation/News/'
# dealsfolder = 'C:/Users/matth/PycharmProjects/Inframation/Global/'

st.header('Deals')
@st.cache_data
def getdata():
    # deals = pd.read_csv(dealsfolder+'deals.csv').iloc[:,4:]
    deals = pd.read_csv('deals.zip').iloc[:,4:]
    # deals['summary.description'] = deals['summary.description'].apply(lambda x: h.handle(str(x)))
    #
    # deals.to_csv(dealsfolder+'deals.csv')

    deals = deals[pd.to_numeric(deals['id'],errors='coerce').notnull()]
    deals.dropna(subset=['nameLowercase','details.description','loanArrangers','lenders','transactionType','dominantRegion','dominantCountry','sectorSortKey','bankdebtsizeUSD','allInvestors'],inplace=True)


    # deals = deals[['nameLowercase','details.description','loanArrangers','transactionType','dominantRegion','dominantCountry','sectorSortKey','bankdebtsizeUSD','allInvestors']]

    deallist = deals['nameLowercase'].unique().tolist()
    deallist.sort()

    return deals,deallist

deals,deallist = getdata()


with st.form('My_form'):
    refdeal = st.selectbox('Select deal',deallist)
    numberrec = st.number_input('Select number of recommendations',value=5,step=1)

    st.form_submit_button('Submit')


region = deals[deals['nameLowercase']==refdeal]['dominantRegion'].iloc[0]
sector = deals[deals['nameLowercase']==refdeal]['sectorSortKey'].iloc[0]
type = deals[deals['nameLowercase']==refdeal]['transactionType'].iloc[0]

deals = deals[(deals['dominantRegion']==region)&(deals['sectorSortKey']==sector)&(deals['transactionType']==type)]

deals.reset_index(inplace=True)


def generate_corpus(details,arrangers,type,region,country,sector,size,sponsors):
    corpus = ''
    arrangers = ' '.join([i for i in eval(arrangers)])
    sponsors = ' '.join(i for i in eval(sponsors))

    corpus+=details+' '+arrangers+' '+type+' '+region+' '+country+' '+sector+' '+str(size)+' mUSD '+sponsors
    return corpus

corpus = []
for i in range(len(deals)):
    corpus.append(generate_corpus(deals['details.description'].iloc[i],deals['loanArrangers'].iloc[i],deals['transactionType'].iloc[i],deals['dominantRegion'].iloc[i],deals['dominantCountry'].iloc[i],deals['sectorSortKey'].iloc[i],deals['bankdebtsizeUSD'].iloc[i],deals['allInvestors'].iloc[i]))


corpus = [corp.replace('"','') for corp in corpus]

deals['corpus']=corpus

# deals = deals[['nameLowercase','corpus']]


tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(deals['corpus'])

cos_mat = linear_kernel(tfidf_matrix,tfidf_matrix)

st.subheader('Recommendations')

def get_recommendations(deal,n):
    index = deals[deals['nameLowercase']==deal].index[0]
    similar_deals = sorted(list(enumerate(cos_mat[index])),reverse=True,key=lambda x: x[1])
    recomm = []
    for i in similar_deals[1:n+1]:
        recomm.append(deals.iloc[i[0]]['nameLowercase'])
    return recomm

target = deals[deals['nameLowercase'].isin(get_recommendations(refdeal,numberrec))]

st.subheader('Deals similar to '+refdeal)

target

st.subheader('Lenders in deals similar to '+refdeal)

target[['nameLowercase','lenders']]
# st.write(get_recommendations(refdeal,numberrec))
