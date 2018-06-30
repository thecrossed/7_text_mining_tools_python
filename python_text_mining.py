
# coding: utf-8
【python Text Mining】7个好玩实用的英文文本挖掘工具实例

如何用计算机帮你计算：

   一篇托福文章有多少个单词？

   其中哪些词互为同义，哪些互为反义？

   标出文章中所有的形容词（副词、连词什么的也行啦）？


   一部《小王子》有多少个句子？

   标出所有句子的主语、谓语？


   一篇Economist读起来难度如何？

   一则川普推文的情感是积极还是消极de？

   一首流行歌曲中，哪些歌词押韵？

   ... ...

作为一名在线英语学习设计师，我每天都要处理、统计、分析大量的陌生英文文本

从打开python的大门起，我就经常探索一些工具来辅助工作：

本文归纳了我用过的工具，如下：

1. nltk(word_tokenize, sent_tokenize, corpus.cmudict, pos_tag)
2. SpaCy
3. textstat
4. textblob

考虑到趣味性，所有的介绍都以解决问题出发，不罗列所有功能

若有兴趣深入研究，请自行进入文档链接1. nltk (word_tokenize, sent_tokenize)NLTK的全称为Natural Language Toolkit，是一套用于英文自然语言处理的Python库与程序。
文档地址：https://www.nltk.org/
NLTK Book 地址：https://www.nltk.org/book/

其中 word_tokenize 和 sent_tokenize 可以对文本分别进行以词、句为单位的切割。
https://www.nltk.org/api/nltk.tokenize.html

问题：比较两篇文章的长度（各自的句子数，各自句子长度）思维步骤：
1. 引入库
2. 调取文本
3. 切割文本（以句子为单位）
4. 计算文本的句子数
5. 切割句子（以单词为单位）
6. 计算句子的单词数# 代码演示
# Step 1 引入库
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize# Step 2 调取文本 text_a, text_b
file_a = open('/Users/jasmine/Desktop/Python_Text_Mining/text_a.txt', 'r')
text_a = file_a.read()
file_b = open('/Users/jasmine/Desktop/Python_Text_Mining/text_b.txt', 'r')
text_b = file_b.read()
#print (text_a)
#print('---------------------')
#print (text_b)# Step 3 切割文本（以句子为单位）
# Step 4 计算文本的句子数
text_a_sents = sent_tokenize(text_a)  # text_a_sents 包含文章的每个句子 （类型：list）
print('Text_a contains',len(text_a_sents),'sentences.') # Text_a  包含的句子数text_b_sents = sent_tokenize(text_b)  # text_b_sents 包含文章的每个句子 （类型：list）
print('Text_b contains',len(text_b_sents),'sentences.') # Text_b  包含的句子数# Step 5 切割句子（以单词为单位）
# Step 6 计算句子的单词数
text_a_count = []
text_b_count = []
for sent in text_a_sents:
    words = word_tokenize(sent) # words 为一个句子包含的单词（包含标点）（类型：list）
    for word in words:    
        if word.isalpha() == False:    # 如果该 word 不是字母（则是标点符号）
            words.remove(word)         # words 剔除该word
    text_a_count.append(len(words))
          
for sent in text_b_sents:
    words = word_tokenize(sent) # words 为一个句子包含的单词（包含标点（类型：list）
    for word in words:    
        if word.isalpha() == False:    # 如果该 word 不是字母（则是标点符号）
            words.remove(word)         # words 剔除该word
    text_b_count.append(len(words))
### 两篇文本长度可视化
import numpy as np
import pandas as pd # pandas
import matplotlib.pyplot as plt # module for plotting 
from matplotlib import gridspec
sentence_no = []
for i in range(len(text_a_count)+1):
    if i != 0:
        sentence_no.append(i)
wc = pd.DataFrame()
wc['sentence_no'] = sentence_no
wc['text_a_count'] = text_a_count
for i in range(19):
    text_b_count.append(0)
wc['text_b_count'] = text_b_count
wc2. nltk(pos_tag)pos_tag 处理一系列的单词，返回单词的词性（part of speech）
https://www.nltk.org/book/ch05.html

问题：如何标出文本中所有de形容词思维步骤：
1. 引入库
2. 调取文本
3. 切割文本 （以句、词为单位）
4. 遍历每个句子中的单词，通过pos_tag返回单词词性
5. 单词词性为“JJ”，放入list adjs 中import nltk                                                                       # Step 1 引入库
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import wordnet

file_adjs = open('/Users/jasmine/Desktop/Python_Text_Mining/text_adjs.txt', 'r')  # Step 2 调取文件
text_adjs = file_adjs.read()

text_adjs_word = word_tokenize(text_adjs)   # Step 3 切割文本 （以句、词为单位）
text_adjs_sent = sent_tokenize(text_adjs)

adjs = []
for sent in text_adjs_sent:          # Step 4 遍历每个句子中的单词，通过pos_tag返回单词词性
    words = word_tokenize(sent)
    pos_word =nltk.pos_tag(words)
    for word in pos_word:            # word类型 为 tuple 
        if word[1] == 'JJ':          # Step 5 单词词性为“JJ”，放入list adjs 中。list of pos tags, see below:
            adjs.append(word[0])     # https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

# In[1]:


### 可视化 

3. nltk.corpus (wordnet)wordnet 是一个巨大的英文单词数据库。名词，动词，形容词，副词等等，在其中都以词义相联系。
http://www.nltk.org/howto/wordnet.html

问题：文章形容词同义标注思维步骤：
1. 引入库
2. 遍历list adjs中的所有单词
3. 从wordnet中，调取该词所有的同义词
4. 如果同义词在list adjs中，并且不是原词
5. printimport nltk                                     # Step 1 引入库
from nltk.corpus import wordnet
syns = {}
for adj in adjs:                                # Step 2 遍历list adjs中的所有单词
    for syn in wordnet.synsets(adj.lower()):    # Step 3 从wordnet中，调取该词所有的同义词
        for l in syn.lemmas():
            if l.name() in adjs:                # Step 4 如果同义词在list adjs中，并且不是原词
                if adj.lower() != l.name():
                    syns.update(((adj, l.name()),))  # Step 5 将单词及其在文章中出现的同义词存入syns4. SpaCySpacy 是一款由Python和Cython写出来的开源库，用于处理NLP任务。相较于NLTK，SpaCy更多应用于工业界。
在之前的任务中，nltk的pos_tag可以为文章中的单词标注词性。现在我们用Spacy尝试将单词在句子中的成分标注出来。
https://spacy.io/

问题：标出文章中所有句子的root verb（谓语动词）#思维步骤：
1. 引入库
2. 加载en_core_web_sm模型
3. 调取文本
4. 遍历每句，引入方法sentence.root import spacy
from spacy.lang.en.examples import sentences                                      # Step 1 引入库
nlp = spacy.load('en_core_web_sm')                                                # Step 2 加载 en_core_web_sm 模型
file_adjs = open('/Users/jasmine/Desktop/Python_Text_Mining/text_adjs.txt', 'r')  # Step 3 调取文本
text_adjs = file_adjs.read()
doc = nlp(text_adjs)                                                              # Step 4 遍历每句，引入方法sentence.root 
for sentence in doc.sents:
    print(sentence, sentence.root)### 可视化5. textstat# textstat 是一款简单好用的文本分析工具，我们可以调用flesch_reading_ease对阅读难易度（readability）分析 
https://pypi.org/project/textstat/问题：特朗普和奥巴马的推文，谁的语言更简单？import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from textstat.textstat import textstat

trump_diff = []
obama_diff = []
file_trump = open('/Users/jasmine/Desktop/Python_Text_Mining/trump_twi.txt', 'r')
trump_twi = file_trump.read()
trump_sent = sent_tokenize(trump_twi)
for sent in trump_sent:
    trump_diff.append(textstat.flesch_reading_ease(sent))
    
file_obama = open('/Users/jasmine/Desktop/Python_Text_Mining/obama_twi.txt', 'r')
obama_twi = file_obama.read()
obama_sent = sent_tokenize(obama_twi)
for sent in obama_sent:
    obama_diff.append(textstat.flesch_reading_ease(sent))
    
### 可视化
import numpy as np
import pandas as pd # pandas
import matplotlib.pyplot as plt # module for plotting 
from matplotlib import gridspec

twi = pd.DataFrame()
for i in range(3):
    obama_diff.append('.')
twi['trump_diff'] = trump_diff
twi['obama_diff'] = obama_diff6. textblobtextblob建立在NLTK和Pattern基础之上，对于刚接触NLP的新手来说，非常友好。
可以做sentiment analysis, pos-tagging, noun phrase extraction
https://textblob.readthedocs.io/en/dev/问题：对Trump和Obama的推文做情感分析import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from textblob import TextBlob

file_trump = open('/Users/jasmine/Desktop/Python_Text_Mining/trump_twi.txt', 'r')
trump_twi = file_trump.read()
trump_sent = sent_tokenize(trump_twi)
trump_senti = []
obama_senti =[]
for sent in trump_sent:
    sent_senti = TextBlob(sent)
    trump_senti.append(sent_senti.sentiment.polarity)

file_obama = open('/Users/jasmine/Desktop/Python_Text_Mining/obama_twi.txt', 'r')
obama_twi = file_obama.read()
obama_sent = sent_tokenize(obama_twi)

for sent in obama_sent:
    sent_senti = TextBlob(sent)
    obama_senti.append(sent_senti.sentiment.polarity)

#print (trump_senti,obama_senti)### 可视化（与readability一起）
for i in range(3):
    obama_senti.append('.')
twi['trump_senti'] = trump_senti
twi['obama_senti'] = obama_senti7. nltk (corpus.cmudict.dict())corpus.cmudict.dict() 是CMU的一套用于NLTK的词典，收录了近13万多的单词含义、发音等信息。
http://www.nltk.org/_modules/nltk/corpus/reader/cmudict.html问题： 一首流行歌曲(Close to you)，那些句子的单词在结尾押韵？思维步骤：
1.import nltk
from nltk.tokenize import word_tokenize,sent_tokenize    
# Step 1 引入库
prondict = nltk.corpus.cmudict.dict()                                               # Step 2 调用corpus.cmudict.dict()
cty_file = open('/Users/jasmine/Desktop/Python_Text_Mining/close_to_you.txt', 'r')  # Step 3 读取文本
cty_text = cty_file.read()
cty_line = cty_text.split('\n')
lines = []
for line in cty_line:
    words = word_tokenize(line)
    for word in words:
        if word.isalpha() ==False:
            words.remove(word)
    lines.append(words)
for line in lines:
    if not line:
        lines.remove(line)
last_words = []
rhymes = []
for line in lines:
    last_word = line[-1]
    last_words.append(last_word)
    last_word=last_word.lower()
    pro_last = prondict[last_word]
    for p in reversed(pro_last[0]):
        if len(p) ==3:
            rhyme = pro_last[0][pro_last[0].index(p):]
            rhymes.append(rhyme)
            #print(rhyme,last_word)
            #print(pro_last[0].index(p),last_word)
            break
### 可视化import numpy as np
import pandas as pd # pandas
import matplotlib.pyplot as plt # module for plotting 
from matplotlib import gridspec

sr = pd.DataFrame()
sr['lines'] =lines
sr['last_word'] = last_words
sr['rhyme'] = rhymes