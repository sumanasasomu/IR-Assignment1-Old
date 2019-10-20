from math import log10, sqrt
from collections import OrderedDict
from flask import Flask, render_template, flash, redirect, url_for, request
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from wtforms import Form, StringField, validators
from csv import reader
import re
from nltk.corpus import stopwords
# nltk.download('stopwords')

app = Flask(__name__)

NUM_DOC = 200
INF = 2**32
songNames = {}
def update_tf_idfTable(doc_stemmedTerms_Table, terms_docList):
	'''returns the tf-idf tdIdfTable'''

	# print(terms_docList)

	''' keys is the list of document IDs '''
	keys = list(doc_stemmedTerms_Table.keys())

	''' N is the No of Documents '''
	N = len(doc_stemmedTerms_Table)

	''' tfIdfTable is the table containing ''' 
	tdIdfTable = []

	''' for each document, i is the current Doc ID '''
	for x in range(N):
		d = keys[x] # i is curr Doc-ID

		''' l is to store the weights of all the terms in docID = i '''
		l = [] 
		for t in terms_docList:

			''' n is the doc freq of term j '''
			n = len(terms_docList[t])

			''' if curr doc has the term j (i.e) id d is in the doclist of term t'''
			if d in terms_docList[t]:

				'''
					W(t,d) = (1+log(tf(t,d)))*log(N/df(t)) 
					tf(t,d) = terms_docList[t][d]
					df(t) = n = len(terms_docList[t])
					
					here term = t and doc = d
				'''

				l.append( ( 1 + log10(terms_docList[t][d]) ) * log10(N/n) )

			else:

				l.append(0)

		''' c = sqrt of ( sum of squares of weights) '''
		c = sqrt(sum(map(lambda x: x*x, l)))

		''' for each term divide its score by sum_of_sqaures_of_scores'''
		for i in range(len(l)):
			l[i] = l[i]/c

		''' store the Total_Score of this particular document '''
		tdIdfTable.append(l)

	# print(tdIdfTable)
	return tdIdfTable

def readDataSet():
	'''to read data from the dataset and load it into Data Structures'''
	print("Reading and processing documents\n")
	f = open('lyrics.csv', 'r', encoding = 'latin-1')
	k = reader(f)

	''' pointing to the row nummber in the dataset '''
	l = 0

	doc_stemmedTerms_Table = {}
	stemmedterms = []
	realTerms = []
	for i in k:
	    l += 1
	    if l == 1:
	    	''' column names '''
	    	continue
	    ''' store DocID '''
	    docId = i[0]
	    songNames[docId] = i[1] 
	    document = i[1] + " " + i[2] + " " + i[4]
	    ''' to remove stop words '''
	    stop_words = set(stopwords.words('english'))
	    word_tokens = word_tokenize(document)
	    filtered_doc = ''
	    for w in word_tokens:
	        if w not in stop_words:
	            filtered_doc += w + " "
	    filtered_doc = filtered_doc.strip()

	    ''' remove numbers and special charcaters'''
	    filtered_doc  = re.sub('[^A-Za-z\s]+', '', filtered_doc) #remove punctuations and spaces and numbers        
	    ''' stemming of words in this document '''        
	    temp_stemmedterms = list(map(lambda x: PorterStemmer().stem(x), word_tokenize(filtered_doc)))

	    temp_realterms = word_tokenize(filtered_doc)

	    ''' store stemmed terms in document docId '''
	    doc_stemmedTerms_Table[docId] = temp_stemmedterms

	    ''' add stemmed words in this doc to the list of all the stemmed words '''
	    stemmedterms += temp_stemmedterms
	    realTerms += temp_realterms

	    ''' stop reading after documents read = NUM_DOC '''
	    if l == NUM_DOC+1:
	    	break

	''' close file '''		
	f.close()
	k = 0
	invertedIndexTable = {}
	
	''' remove duplicate stemmed words (which are in the corpus) '''
	stemmedterms = list(set(stemmedterms))
	realTerms = set(realTerms)

	''' for each token in corpus '''
	for i in stemmedterms:
		''' create a posting list for this token '''
		postingsList = []

		'''
			check if the token is present in the document j
			if yes? append the document and the term frequency to Postings list
		'''
		for j in doc_stemmedTerms_Table:
			if i in doc_stemmedTerms_Table[j]:
				postingsList.append((j, doc_stemmedTerms_Table[j].count(i)))

		k += 1

		''' make the postings list as a dictionary and update the inverted index table '''
		invertedIndexTable[i] = dict(postingsList)

	# print(invertedIndexTable)
	# print(doc_stemmedTerms_Table)

	''' document vs tokens table is sorted according to the document ID '''
	doc_stemmedTerms_Table = OrderedDict(doc_stemmedTerms_Table)

	''' tdIdfTable is returned by the function update_tf_idfTable '''
	tdIdfTable = update_tf_idfTable(doc_stemmedTerms_Table, invertedIndexTable)

	''' return doc vs. tokens table & InvertedIndex Table & TfIdfTable & actual tokens in corpus'''
	return doc_stemmedTerms_Table, invertedIndexTable, tdIdfTable, realTerms


doc_stemmedTerms_Table, invertedIndexTable, tdIdfTable, real_terms = readDataSet()

''' N = no of documents '''
N = len(doc_stemmedTerms_Table)

def editDistance(s1, s2, m,  n):
	'''determines the minimum number operations to convert s1 s2 by using dynamic programming'''

	'''create a matrix of size m by n [m,n] '''
	dp = [[0 for x in range(n+1)] for x in range(m+1)] 
	for i in range(m+1):
		for j in range(n+1):
			if i==0:
				dp[i][j] = j
			elif j==0:
				dp[i][j] = i
			elif s1[i-1]==s2[j-1]:
				dp[i][j] = dp[i-1][j-1]
			else:
				''' if element is to be added or replaced or deleted '''
				dp[i][j] = 1 + min(dp[i][j-1], dp[i-1][j-1], dp[i-1][j])
	
	return dp[m][n]

def getNearestTerm(rterms, word):
	'''returns a term in terms which requires least number of operations to convert to given word'''
	minimum = INF
	minWord = word
	for i in rterms:
		k = editDistance(i, word, len(i), len(word))
		if k < minimum:
			minWord = i
			minimum = k
	return minWord


def getTermWithSuffix(rterms, word):
	'''returns a term from real terms whose suffix is the given word'''
	for i in rterms:
		''' if the word is in the set of real terms then returns true and we then add the real term to the query '''
		if i.find(word)==0:
			return True, i
	''' if there is no real term with suffix as word then add the query word as it is to the query'''
	return False, word

def getResults(query):
	'''relevant documents for the query entered are returned'''
	search = ''
	stop_words = set(stopwords.words('english'))
	word_tokens = word_tokenize(query)
	filtered_query = ''
	for w in word_tokens:
		if w not in stop_words:
			filtered_query += w + " "

	filtered_query = filtered_query.strip()
	''' remove numbers and special charcaters'''
	filtered_query = re.sub('[^A-Za-z\s]+', '', filtered_query)
	query = filtered_query
	for i in query.split(' '):
		if PorterStemmer().stem(i) in invertedIndexTable:
			search += i + ' '
		else:
			result, word = getTermWithSuffix(real_terms, i)
			''' if you can find the real term with i from query as suffix then add the real term'''
			if result == True:
				search += word + ' '
				''' if no such word then add nearest term to the query '''
			else:
				search += getNearestTerm(real_terms, i) + ' '

	search = search.strip()
	queryTermsList = list(map(PorterStemmer().stem , word_tokenize(search)))

	''' form a vector for query '''
	''' q is weights of all the terms in given query '''
	q = []
	for i in invertedIndexTable:
		n = len(invertedIndexTable[i])
		''' N = no of documents '''
		tf = queryTermsList.count(i)
		if tf>0:
			q.append((1+log10(tf)) * log10((N+1)/(n+1)))
		else:
			q.append(0)

	c = sqrt(sum(map(lambda x: x*x, q)))
	for i in range(len(q)):
		q[i] = q[i]/c

	def getDotProduct(a, b):
		'''returns the dot product'''
		su = 0
		for i in range(len(a)):
			su += a[i] * b[i]
		return su

	''' ranking of documents and return result '''
	ranks = {}
	ranks2 = {}
	k = 0
	documents_list = list(doc_stemmedTerms_Table.keys())
	for j in tdIdfTable:
		ranks2[songNames[documents_list[k]]] = getDotProduct(j, q)
		ranks[documents_list[k]] = getDotProduct(j, q)
		k += 1
	results2 = sorted(ranks2, key = lambda x: ranks2[x], reverse = True)[:10]
	results = sorted(ranks, key = lambda x: ranks[x], reverse = True)[:10]
	print(results)
	return results,results2, search	


def fetchDocumentDetails(doc_id):
	'''document of docID doc_id is returned'''
	f = open('lyrics.csv', 'r', encoding = 'latin-1')
	k = reader(f)
	l = 0
	for i in k:
		l += 1
		if l == 1:
			continue
		docId = i[0]
		if doc_id == docId:
			f.close()
			return i[1], i[2], i[4]



class searchBar(Form):
	'''for form in home page'''
	search = StringField('Search Lyrics: ', [validators.InputRequired()])

@app.route('/', methods = ['GET', 'POST'])
def index():
	'''home page'''
	form = searchBar(request.form)
	if request.method == 'POST' and form.validate():
		search = form.search.data
		return redirect(url_for('searchResults', query = search, form = form))
	return render_template("home.html", form = form)

@app.route('/searchResults/<string:query>', methods = ['GET', 'POST'])
def searchResults(query):
	'''to display top 10 relevant documents'''
	results,results2, real_search = getResults(query)
	form = searchBar(request.form)
	if request.method == 'POST' and form.validate():
		search = form.search.data
		return redirect(url_for('searchResults', query = search))
	if query != real_search:
		flash('Showing results for ' + real_search, 'success')
	return render_template('searchResults.html', results = results, form = form)

@app.route('/displayDoc/<string:docname>')
def displayDoc(docname):
	'''to display the document with given doc id'''
	Song, Artist, Lyrics = fetchDocumentDetails(docname)
	return render_template('document.html', Artist = Artist.capitalize(), Song = Song.upper(), Lyrics = Lyrics.capitalize())


if __name__ == '__main__':
	app.secret_key = 'development key'
	app.run()

'''
start_time = tm.time()
print("--- %s seconds ---" % (tm.time() - start_time))
'''
