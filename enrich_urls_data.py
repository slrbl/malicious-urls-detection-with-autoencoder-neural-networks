# Version: 1.0 - 2018/06/29
# Contact: walid.daboubi@gmail.com

import sys
import os
import csv

LETTERS = 'abcdefghijklmnopqrstuwxyz'
NUMBERS = '0123456789'
SPEC_CHARS = ['+','\"','*','#','%','&','(',')','=','?','^','-','.','!','~','_','>','<']

enriched_csv = open('url_enriched_data.csv', 'w')
enriched_csv.write('len,spec_chars,domain,depth,numericals_count,word_count,label\n')

def check_url_contains_words(url):
    found_words = []
    for letter in LETTERS:
        dictionary = open('{}/dictionary/wb1913_{}.txt'.format(os.path.dirname(os.path.realpath(__file__)), letter), 'r')
        for line in dictionary.readlines():
            word = line.split('</B>')[0].replace('<P><B>', '').lower()
            if str(word) in url.lower() and len(word) > 1 and word not in found_words:
                found_words.append(word)
    return len(found_words)

count = 0

for row in  csv.reader(open('url_data.csv', 'r'), delimiter = ','):
    print str(count)
    count += 1
    if 'bad' in row[1].lower():
        label = '1'
    else:
        label = '0'
    spec_chars = 0
    depth = 0
    numericals_count = 0
    word_count = 0
    url = str(l[0])
    #print url
    word_count = check_url_contains_words(url)
    for c in str(l):
        if c in SPEC_CHARS:
            spec_chars += 1
        if c in ['/']:
            depth += 1
        if c in NUMBERS:
            numericals_count += 1
    enriched_csv.write(str(len(l[0])) + ',' + str(spec_chars) + ',0,' + str(depth) + ',' + str(numericals_count) + ',' + str(word_count) + ',' + label + '\n')
