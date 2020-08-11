import json
import urllib.parse
import urllib.request
import sys
import time
import os
from tqdm import tqdm
from collections import defaultdict

class WikiPedia():
    def __init__(self, title_file):
        self.title_file = title_file
        self.titles = []
        with open(title_file) as f:
            for i, line in enumerate(f):
                title = line.strip(' \n').split(' ')[-1]
                self.titles.append(title)
    
    def get(self):

        articles_dict=dict()
        for title in tqdm(self.titles, desc='get articles'):
            url = "https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exlimit=max&explaintext&redirects&titles="+urllib.parse.quote_plus(title)
            json_doc = urllib.request.urlopen(url).read().decode("utf-8", errors="ignore")
            parsed = json.loads(json_doc)
            pages = parsed["query"]["pages"]

            for i in pages:
                page = pages[i]
                try:
                    title=page["title"].encode(encoding="utf-8", errors="ignore").decode(encoding="utf-8")
                    content=page["extract"].encode(encoding="utf-8", errors="ignore").decode(encoding="utf-8")
                    articles_dict[title]=content
                except:
                    print ('error : %s'%title)
                    print ('URL : %s'%url)
            if len(articles_dict) >= 10000:
                break
        return articles_dict

    def crawl(self,):
        articles = self.get()
        
        result = []
        for title, article in tqdm(articles.items(), desc='parse articles'):
            article = article.split('\n')
            paragraphs = []
            for para in article :
                if len(para)>500 and len(para)<3500:
                    paragraphs.append({'context':para})
        
            if paragraphs:
                result.append({'title':title, 'paragraphs':paragraphs})

        _, filename = os.path.split(self.title_file)

        with open('data/crawl/%s.txt'%filename.split('.')[0], 'w') as f:
            json.dump(result, f, indent=1, ensure_ascii=False)

    def remove_squad_title(self, squad_train_path, squad_dev_path):
        squad_title={}
        def get_title(path, squad_title):
            with open(path, 'r')  as f:
                data = json.load(f)
                for elem in data['data']:
                    title = elem['title']
                    squad_title[title]=1
        get_title(squad_train_path, squad_title)
        get_title(squad_dev_path, squad_title)

        filtered_titles = []
        for t in self.titles:
            if t not in squad_title:
                filtered_titles.append(t)
        self.title = filtered_titles

if __name__== '__main__':
    print (sys.argv)
    title_file = sys.argv[1]
    squad_train_path = sys.argv[2]
    squad_dev_path = sys.argv[3]
    w = WikiPedia(title_file)
    w.remove_squad_title(squad_train_path, squad_dev_path)
    w.crawl()
