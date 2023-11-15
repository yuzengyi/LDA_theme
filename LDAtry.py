from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Define the comments to be analyzed
comments = [
    "真实模拟课堂氛围，学生真的有回答发言。可能有些许不流畅。从教学法的角度，运用纸贴的方式，表示流程图的绘制过程，很新颖，但最好上面不是空白的，就是有字，可能会更好。",
    "讲的很好。要是对学生的回答反馈再多一点就更好了。之前上学期在演练时老师好像说不要只是说同学们回答的非常好，最好提一下同学们都说了些什么。以及最后多元化评价那里最好加一个权重。",
    "教态自然，PPT的重点突出。可能例子太多，例子之间的逻辑性可以考虑一下~ 教学方法、核心素养合理适当。",
    "昊哥教态自然，一点都不拘束，也很幽默。好像会经常说“那么”？我也经常会说“因此”，就感觉可能会是口头禅。钲凯哥声音很好听，如果能在面朝着我们，少看PPT就更好啦。彦杰哥教态很好，说话也很清楚。教学方法、核心素养合理适当。",
    "讲述法是不错的，可能时间有点长了。互动与反馈可以增加。",
    "举的例子不错!开头导入也很好。板书设计有逻辑性。"
]

# Instantiate the Count Vectorizer
vectorizer = CountVectorizer(stop_words=['的', '了', '在', '是', '和', '也', '有'])

# Convert the documents into a document-term matrix
X = vectorizer.fit_transform(comments)

# Instantiate the LDA model
lda = LatentDirichletAllocation(n_components=4, random_state=0)

# Fit the LDA model to the document-term matrix
lda.fit(X)

# Function to print top words for each topic
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)

# Call this function to print the top words
print_top_words(lda, vectorizer.get_feature_names_out(), 10)
