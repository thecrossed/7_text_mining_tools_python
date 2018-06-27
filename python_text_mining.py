
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
2. Spacy
3. textstat
4. textblob

考虑到趣味性，所有的介绍都以解决问题出发，不罗列各种功能

若有兴趣深入研究，请自行进入文档链接1. nltk (word_tokenize, sent_tokenize)介绍nltk(word_tokenize, sent_tokenize)# 计算文章单词数，句子数／ 计算句子单词数/ 计算独立单词数  | 问题：比较两篇文章的长度分布（句子数，句子长度）
# 先看一下文本
file = open('/Users/jasmine/Desktop/the_little_prince.txt', 'r')
text = file.read()
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
text_word = word_tokenize(text)
### 删标点，小写字母化，独立单词计算
text_sent = sent_tokenize(text)
### 计算每句的单词数，histo句子单词数2. nltk(pos_tag)
# 词性分析 | 问题：标出所有形容词
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize

file = open('/Users/jasmine/Desktop/the_little_prince.txt', 'r')
text = file.read()
text_word = word_tokenize(text)
text_sent = sent_tokenize(text)
pos_word = nltk.pos_tag(text_word)
### 可视化

CC	coordinating conjunction
CD	cardinal digit
DT	determiner
EX	existential there (like: "there is" ... think of it like "there exists")
FW	foreign word
IN	preposition/subordinating conjunction
JJ	adjective	'big'
JJR	adjective, comparative	'bigger'
JJS	adjective, superlative	'biggest'
LS	list marker	1)
MD	modal	could, will
NN	noun, singular 'desk'
NNS	noun plural	'desks'
NNP	proper noun, singular	'Harrison'
NNPS	proper noun, plural	'Americans'
PDT	predeterminer	'all the kids'
POS	possessive ending	parent's
PRP	personal pronoun	I, he, she
PRP$	possessive pronoun	my, his, hers
RB	adverb	very, silently,
RBR	adverb, comparative	better
RBS	adverb, superlative	best
RP	particle	give up
TO	to	go 'to' the store.
UH	interjection	errrrrrrrm
VB	verb, base form	take
VBD	verb, past tense	took
VBG	verb, gerund/present participle	taking
VBN	verb, past participle	taken
VBP	verb, sing. present, non-3d	take
VBZ	verb, 3rd person sing. present	takes
WDT	wh-determiner	which
WP	wh-pronoun	who, what
WP$	possessive wh-pronoun	whose
WRB	wh-abverb	where, when3. nltk.corpus (wordnet)# 找出一篇作文中paraphrase的形容词 | 问题：官方范文形容词同义标注
import nltk
from nltk.corpus import wordnet
synonyms = []
antonyms = []
 
for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
 
print(set(synonyms))
print(set(antonyms))4. 找句子主语 Spacy# 找句子的主语和谓语。| 问题：标出所有句子的主语，对应的谓语。
import spacy
from spacy.lang.en.examples import sentences

nlp = spacy.load('en_core_web_sm')
doc = nlp('Once when I was six years old I saw a magnificent picture in a book, called True Stories from Nature, about the primeval forest.')
print(doc.text)
for token in doc:
    print(token.text, token.pos_, token.dep_)5. textstat
# 阅读难易度分析 readability | 问题：一篇未知的经济学人文章，难度如何，和某篇托福文章比较
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from textstat.textstat import textstat

file = open('/Users/jasmine/Desktop/the_little_prince.txt', 'r')
text = file.read()
text_sent = sent_tokenize(text)
for sent in text_sent:
    print (textstat.flesch_reading_ease(sent))
    print(sent)

### 可视化6. textblob
# 先看一下文本 情感分析 | 一篇GRE阅读的观点走向
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from textblob import TextBlob

file = open('/Users/jasmine/Desktop/the_little_prince.txt', 'r')
text = file.read()
text_sent = sent_tokenize(text)
for sent in text_sent:
    sent_senti = TextBlob(sent)
    print(sent,sent_senti.sentiment.polarity)
### 可视化7. nltk (corpus.cmudict.dict())
# 音节音素分析 | 一首流行歌曲，每句结尾的单词，哪些押韵
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
entries = nltk.corpus.cmudict.entries()
prondict = nltk.corpus.cmudict.dict()
file = open('/Users/jasmine/Desktop/trouble.txt', 'r')
text = file.read()
text_sent = sent_tokenize(text)
for line in text_sent:
    word = word_tokenize(line)
    for w in word:
        w=w.lower()
        try:
            print (w,prondict[w])
        except:
            print(w)
            continue5. gender_guesserimport gender_guesser.detector as gender
# 性别比例
gender_speaker = []
names = []
for name in title_rank['main_speaker']:
    name = name.split()[0]
    names.append(name)
d = gender.Detector()
for n in names:
    gender_speaker.append(d.get_gender(n))
title_rank['gender_speaker'] = gender_speaker
title_rank.head(100)
gender_count = title_rank['gender_speaker'].value_counts()
gender_count结尾：如何获取文本数据？https://www.r-bloggers.com/text-mining-in-r-and-python-8-tips-to-get-started/