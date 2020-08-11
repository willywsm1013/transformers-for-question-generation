# use  Project Nayuki’s Wikipedia’s internal PageRanks (https://www.nayuki.io/page/computing-wikipedias-internal-pageranks)
# to obtain top 10000 articles.

squad_path=../squad

#  step 1: download English Wikipedia data dumps from  https://dumps.wikimedia.org/enwiki/
#          From a snapshot of your choice, download the two files enwiki-yyyymmdd-page.sql.gz (~1 GB) 
#          and enwiki-yyyymmdd-pagelinks.sql.gz (~5 GB). Put these two files into ./pagerank. 
#          Update the file names at the top of ./pagerank/WikipediaPagerank.java and compile. 


# step 2: run pagerank and get titles
cd pagerank
javac WikipediaPagerank.java
java -mx15G WikipediaPagerank 
cd ../

# step 3: crawl articles from wikipedia and remove titiles appear in squad training set and dev set
title_file=pagerank/wiki-20220401.txt
python crawl_wikipedia.py ${title_file} ${squad_path}/train.json ${squad_path}/dev.json

